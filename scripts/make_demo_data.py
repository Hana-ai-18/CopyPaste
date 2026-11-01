"""Generate a small DEMO split (3 fake businesses: csv / xlsx / sqlite) with gold answers and scripted
LLM responses, to smoke-test the whole pipeline WITHOUT a GPU, a model or the real competition data.

    python scripts/make_demo_data.py --out demo_split

Gold answers are computed with plain python loops, independently of the pandas helpers used by the
scripted solutions, so a passing smoke test really exercises loader + helpers + sandbox + voting + scoring.
The lunar calendar written here is a SYNTHETIC approximation (only Tet dates are real); it is for tests only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sqlite3
import textwrap
from collections import defaultdict
from pathlib import Path

import pandas as pd

AS_OF = dt.date(2026, 3, 4)  # a Wednesday

RULEBOOK = """# Rulebook (DEMO split)
- Revenue = sum of line amounts (qty * unit_price - line_discount). Order-level vouchers are NOT deducted. VAT and delivery fees are excluded.
- Cancelled orders and orders with any returned item (returned_qty > 0) are excluded.
- "ban chay" (best selling) is ranked by quantity.
- A week runs Monday to Sunday. "tuan nay" = Monday of the as-of week up to the as-of date. "tuan roi" = the previous full week.
- "thang nay" = the month containing the as-of date, up to and including that date. "thang truoc" = the previous full month.
- Default cost = weighted-average purchase cost by lot.
- Tet window = 23 thang Chap to mung 10 (lunar calendar), see calendar.csv.
- Rankings: descending by the criterion; equal values ordered by code ascending.
"""

PRODUCTS = [("Sting dâu 330ml", "Nước ngọt", 10000), ("Nước suối Aquafina 500ml", "Nước", 6000), ("Mì Hảo Hảo tôm chua cay", "Mì", 4500),
            ("Sữa Vinamilk 180ml", "Sữa", 8000), ("Bánh Oreo", "Bánh", 12000), ("Coca Cola lon 330ml", "Nước ngọt", 11000),
            ("Dầu ăn Neptune 1L", "Gia vị", 52000), ("Gạo ST25 5kg", "Gạo", 120000), ("Nước mắm Nam Ngư 500ml", "Gia vị", 38000),
            ("Đường Biên Hòa 1kg", "Gia vị", 24000), ("Trứng gà vỉ 10", "Thực phẩm", 35000), ("Khăn giấy Pulppy", "Gia dụng", 15000)]
CUSTOMERS = [("Chị Lan", 3200000), ("Anh Tú", 1500000), ("Cô Hạnh", 0), ("Chú Bình", 870000), ("Bé Na", 0), ("Anh Khoa", 0), ("Chị Mai", 2450000), ("Ông Sáu", 0)]


# ------------------------------------------------------------------ business 1: grocery (csv)
def build_grocery(seed: int):
    rng = random.Random(seed)
    products = [{"product_id": f"SP{i + 1:04d}", "product_name": n, "category": c, "price": p} for i, (n, c, p) in enumerate(PRODUCTS)]
    weights = [rng.uniform(0.5, 3.0) for _ in products]
    customers = [{"customer_id": f"KH{i + 1:03d}", "customer_name": n, "phone": f"09{rng.randint(10000000, 99999999)}", "debt": d} for i, (n, d) in enumerate(CUSTOMERS)]
    orders, items = [], []
    day, n = dt.date(2025, 1, 1), 0
    while day <= AS_OF:
        for _ in range(rng.randint(2, 5)):
            n += 1
            oid = f"DH{n:06d}"
            status = "cancelled" if rng.random() < 0.08 else "completed"
            orders.append({"order_id": oid, "order_date": day.isoformat(), "customer_id": rng.choice(customers)["customer_id"], "status": status,
                           "voucher_discount": rng.choice([0, 0, 5000]), "shipping_fee": rng.choice([0, 0, 15000])})
            for p in rng.sample(range(len(products)), rng.randint(1, 4)):
                qty = rng.randint(1, 6)
                price = products[p]["price"]
                disc = rng.choice([0, 0, round(qty * price * 0.05)])
                items.append({"order_id": oid, "product_id": products[p]["product_id"], "qty": qty, "unit_price": price, "line_discount": disc,
                              "returned_qty": 1 if rng.random() < 0.04 else 0})
        day += dt.timedelta(days=1)
    return products, customers, orders, items, weights


def gold_grocery(products, customers, orders, items):
    """Gold answers via plain python (independent of pandas helpers). Returns None on ties."""
    pname = {p["product_id"]: p["product_name"] for p in products}
    bad = {it["order_id"] for it in items if it["returned_qty"] > 0}
    odate = {o["order_id"]: dt.date.fromisoformat(o["order_date"]) for o in orders if o["status"] != "cancelled" and o["order_id"] not in bad}

    def lines(s, e):
        return [(it, odate[it["order_id"]]) for it in items if it["order_id"] in odate and s <= odate[it["order_id"]] <= e]

    def revenue(s, e):
        return sum(it["qty"] * it["unit_price"] - it["line_discount"] for it, _ in lines(s, e))

    def top_by(s, e, key):
        agg = defaultdict(int)
        for it, _ in lines(s, e):
            agg[it["product_id"]] += key(it)
        rank = sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))
        return rank

    monday = AS_OF - dt.timedelta(days=AS_OF.weekday())
    lw_s, lw_e = monday - dt.timedelta(days=7), monday - dt.timedelta(days=1)
    pw_s, pw_e = lw_s - dt.timedelta(days=7), lw_s - dt.timedelta(days=1)
    mtd_s = AS_OF.replace(day=1)
    feb_s, feb_e = dt.date(2026, 2, 1), dt.date(2026, 2, 28)
    jan_s, jan_e = dt.date(2026, 1, 1), dt.date(2026, 1, 31)

    out = {}
    r = top_by(lw_s, lw_e, lambda it: it["qty"])
    if len(r) < 2 or r[0][1] == r[1][1]:
        return None
    out["best_last_week"] = (pname[r[0][0]], [r[0][0]])
    out["rev_mtd"] = revenue(mtd_s, AS_OF)
    jan, feb = revenue(jan_s, jan_e), revenue(feb_s, feb_e)
    out["pct_feb_vs_jan"] = round((feb - jan) / jan * 100, 2)
    r = top_by(feb_s, feb_e, lambda it: it["qty"] * it["unit_price"] - it["line_discount"])
    if r[2][1] == r[3][1]:
        return None
    out["top3_rev_feb"] = [pname[k] for k, _ in r[:3]]
    perday = defaultdict(int)
    for it, d in lines(feb_s, feb_e):
        perday[d] += it["qty"] * it["unit_price"] - it["line_discount"]
    rank = sorted(perday.items(), key=lambda kv: (-kv[1], kv[0]))
    if rank[0][1] == rank[1][1]:
        return None
    out["best_day_feb"] = rank[0][0].isoformat()
    a, b = revenue(lw_s, lw_e), revenue(pw_s, pw_e)
    if a == b:
        return None
    out["lastweek_gt_prevweek"] = a > b
    debts = sorted([(c["debt"], c["customer_id"], c["customer_name"]) for c in customers if c["debt"] > 0], key=lambda t: (-t[0], t[1]))
    out["debt_table"] = {"columns": ["customer_name", "debt"], "rows": [[n, d] for d, _, n in debts]}
    out["rev_yesterday"] = revenue(AS_OF - dt.timedelta(days=1), AS_OF - dt.timedelta(days=1))
    return out


# ------------------------------------------------------------------ business 2: cafe (xlsx with junk title rows)
def build_cafe(seed: int):
    rng = random.Random(seed)
    menu = [{"item_code": f"M{i + 1:02d}", "item_name": n, "price": p} for i, (n, p) in enumerate(
        [("Cà phê sữa", 29000), ("Bạc xỉu", 32000), ("Trà đào", 35000), ("Trà sữa trân châu", 38000), ("Sinh tố bơ", 42000), ("Nước cam", 30000)])]
    sales, d = [], dt.date(2026, 1, 1)
    while d <= AS_OF:
        for m in menu:
            cups = rng.randint(2, 30)
            sales.append({"sale_date": dt.datetime(d.year, d.month, d.day), "item_code": m["item_code"], "cups": cups, "amount": cups * m["price"]})
        d += dt.timedelta(days=1)
    return menu, sales


def gold_cafe(menu, sales):
    name = {m["item_code"]: m["item_name"] for m in menu}
    monday = AS_OF - dt.timedelta(days=AS_OF.weekday())
    cups_week = sum(s["cups"] for s in sales if monday <= s["sale_date"].date() <= AS_OF)
    agg = defaultdict(int)
    for s in sales:
        if AS_OF.replace(day=1) <= s["sale_date"].date() <= AS_OF:
            agg[s["item_code"]] += s["cups"]
    rank = sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))
    if rank[0][1] == rank[1][1]:
        return None
    return {"cups_week": cups_week, "best_item_month": (name[rank[0][0]], [rank[0][0]])}


# ------------------------------------------------------------------ business 3: pharmacy (sqlite)
def build_pharmacy(seed: int):
    rng = random.Random(seed)
    meds = [{"med_id": f"T{i + 1:03d}", "med_name": n, "unit": "hộp"} for i, n in enumerate(
        ["Paracetamol 500mg", "Amoxicillin 500mg", "Vitamin C 1000mg", "Cetirizine 10mg", "Omeprazole 20mg", "Berberin", "Oresol", "Panadol Extra", "Efferalgan", "Alpha Choay"])]
    batches = []
    for i in range(26):
        m = meds[i % len(meds)]
        exp = AS_OF + dt.timedelta(days=rng.randint(-200, 400))
        batches.append({"batch_id": f"L{i + 1:03d}", "med_id": m["med_id"], "expiry_date": exp.isoformat(), "stock_qty": rng.choice([0, 0, 5, 12, 30, 48])})
    return meds, batches


def gold_pharmacy(meds, batches):
    name = {m["med_id"]: m["med_name"] for m in meds}
    expired = {b["med_id"] for b in batches if b["stock_qty"] > 0 and dt.date.fromisoformat(b["expiry_date"]) < AS_OF}
    soon = defaultdict(lambda: dt.date.max)
    for b in batches:
        e = dt.date.fromisoformat(b["expiry_date"])
        if b["stock_qty"] > 0 and AS_OF <= e <= AS_OF + dt.timedelta(days=30):
            soon[b["med_id"]] = min(soon[b["med_id"]], e)
    rank = sorted(soon.items(), key=lambda kv: (kv[1], kv[0]))
    return {"n_expired": len(expired), "soon": [name[k] for k, _ in rank]}


# ------------------------------------------------------------------ synthetic lunar calendar (TEST ONLY)
def build_calendar() -> pd.DataFrame:
    tet = [dt.date(2024, 2, 10), dt.date(2025, 1, 29), dt.date(2026, 2, 17), dt.date(2027, 2, 6)]
    rows = []
    for ly in range(3):
        start, nxt = tet[ly], tet[ly + 1]
        total = (nxt - start).days
        lens = [30 if i % 2 == 0 else 29 for i in range(12)]
        months = [(i + 1, False, lens[i]) for i in range(12)]
        extra = total - sum(lens)
        if extra > 0:
            months.insert(6, (6, True, extra))
        day = start
        for m, leap, ln in months:
            for dd in range(1, ln + 1):
                rows.append({"date": day.isoformat(), "lunar_year": 2024 + ly, "lunar_month": m, "lunar_day": dd, "is_leap": int(leap)})
                day += dt.timedelta(days=1)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ scripted solutions (what a good LLM would write)
def D(s: str) -> str:
    return textwrap.dedent(s).strip("\n")


VALID = D("""
    bad = set(order_items.loc[order_items.returned_qty > 0, 'order_id'])
    ok_orders = orders[(orders.status != 'cancelled') & ~orders.order_id.isin(bad)]
    def rev(s, e):
        o = ok_orders[between(ok_orders.order_date, s, e)]
        li = order_items[order_items.order_id.isin(o.order_id)]
        return (li.qty * li.unit_price - li.line_discount).sum()
""")

SOL = {
    "best_last_week": VALID + "\n" + D("""
        s, e = last_week(as_of)
        o = ok_orders[between(ok_orders.order_date, s, e)]
        li = order_items[order_items.order_id.isin(o.order_id)]
        qty = li.groupby('product_id', as_index=False).qty.sum().merge(products[['product_id', 'product_name']], on='product_id')
        top = rank_top(qty, 'qty', 'product_id', 1)
        answer_type = 'string'
        answer = top.product_name.iloc[0]
    """),
    "rev_mtd": VALID + "\n" + D("""
        s, e = this_month_to_date(as_of)
        answer_type = 'number'
        answer = round_vnd(rev(s, e))
    """),
    "pct": VALID + "\n" + D("""
        feb = rev(*month_range(2026, 2))
        jan = rev(*month_range(2026, 1))
        answer_type = 'percent'
        answer = pct_change(feb, jan)
    """),
    "top3": VALID + "\n" + D("""
        s, e = last_month(as_of)
        o = ok_orders[between(ok_orders.order_date, s, e)]
        li = order_items[order_items.order_id.isin(o.order_id)].copy()
        li['amount'] = li.qty * li.unit_price - li.line_discount
        g = li.groupby('product_id', as_index=False).amount.sum().merge(products[['product_id', 'product_name']], on='product_id')
        top = rank_top(g, 'amount', 'product_id', 3)
        answer_type = 'list'
        answer = top.product_name.tolist()
    """),
    "best_day": VALID + "\n" + D("""
        s, e = last_month(as_of)
        o = ok_orders[between(ok_orders.order_date, s, e)]
        li = order_items.merge(o[['order_id', 'order_date']], on='order_id')
        li['amount'] = li.qty * li.unit_price - li.line_discount
        d = li.groupby('order_date', as_index=False).amount.sum()
        top = rank_top(d, 'amount', 'order_date', 1)
        answer_type = 'date'
        answer = top.order_date.iloc[0]
    """),
    "week_cmp": VALID + "\n" + D("""
        s1, e1 = last_week(as_of)
        s0, e0 = last_week(s1)
        answer_type = 'boolean'
        answer = bool(rev(s1, e1) > rev(s0, e0))
    """),
    "debt": D("""
        d = customers[customers.debt > 0].sort_values(['debt', 'customer_id'], ascending=[False, True])
        answer_type = 'table'
        answer = {'columns': ['customer_name', 'debt'], 'rows': d[['customer_name', 'debt']].values.tolist()}
    """),
    "profit": "declare_unanswerable('Dữ liệu không có giá vốn nên không tính được lợi nhuận')",
    "cups_week": D("""
        s, e = this_week_to_date(as_of)
        answer_type = 'number'
        answer = round_vnd(sales.loc[between(sales.sale_date, s, e), 'cups'].sum())
    """),
    "best_item": D("""
        s, e = this_month_to_date(as_of)
        g = sales[between(sales.sale_date, s, e)].groupby('item_code', as_index=False).cups.sum().merge(menu[['item_code', 'item_name']], on='item_code')
        top = rank_top(g, 'cups', 'item_code', 1)
        answer_type = 'string'
        answer = top.item_name.iloc[0]
    """),
    "n_expired": D("""
        b = batches.copy()
        b['exp'] = parse_dates(b.expiry_date)
        exp = b[(b.exp < pd.Timestamp(as_of)) & (b.stock_qty > 0)]
        answer_type = 'number'
        answer = int(exp.med_id.nunique())
    """),
    "soon": D("""
        b = batches[batches.stock_qty > 0].copy()
        b['exp'] = parse_dates(b.expiry_date)
        w = b[(b.exp >= pd.Timestamp(as_of)) & (b.exp <= pd.Timestamp(as_of) + pd.Timedelta(days=30))]
        g = w.groupby('med_id', as_index=False).exp.min().merge(medicines, on='med_id').sort_values(['exp', 'med_id'])
        answer_type = 'list'
        answer = g.med_name.tolist()
    """),
}
BROKEN_KEY = SOL["best_last_week"].replace("'product_name'", "'ten_san_pham'", 1)   # KeyError -> must be repaired
WRONG_PCT = SOL["pct"].replace(  # forgets to exclude orders with returned items
    "ok_orders = orders[(orders.status != 'cancelled') & ~orders.order_id.isin(bad)]", "ok_orders = orders[(orders.status != 'cancelled')]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_split")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)

    for seed in range(1, 200):
        products, customers, orders, items, _ = build_grocery(seed)
        g1 = gold_grocery(products, customers, orders, items)
        if g1:
            break
    else:
        raise SystemExit("no tie-free grocery seed found")
    b1 = out / "data" / "biz_demo1"
    b1.mkdir(parents=True, exist_ok=True)
    for name, rows in [("products", products), ("customers", customers), ("orders", orders), ("order_items", items)]:
        pd.DataFrame(rows).to_csv(b1 / f"{name}.csv", index=False, encoding="utf-8-sig")
    (b1 / "README.md").write_text("Tiệm tạp hóa nhỏ (DEMO). products: danh mục hàng; orders/order_items: đơn và chi tiết đơn; customers: khách quen, cột debt là công nợ.\n", encoding="utf-8")

    for seed in range(1, 200):
        menu, sales = build_cafe(seed)
        g2 = gold_cafe(menu, sales)
        if g2:
            break
    b2 = out / "data" / "biz_demo2"
    b2.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(b2 / "quan_cafe.xlsx", engine="openpyxl") as w:
        pd.DataFrame(menu).to_excel(w, sheet_name="menu", index=False)
        pd.DataFrame(sales).to_excel(w, sheet_name="sales", index=False, startrow=2)
        w.sheets["sales"]["A1"] = "Báo cáo bán hàng quý 1/2026"
    (b2 / "README.md").write_text("Quán cà phê (DEMO). Sheet sales có 2 dòng tiêu đề thừa phía trên.\n", encoding="utf-8")

    meds, batches = build_pharmacy(3)
    g3 = gold_pharmacy(meds, batches)
    b3 = out / "data" / "biz_demo3"
    b3.mkdir(parents=True, exist_ok=True)
    db = b3 / "nha_thuoc.sqlite"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    pd.DataFrame(meds).to_sql("medicines", con, index=False)
    pd.DataFrame(batches).to_sql("batches", con, index=False)
    con.close()
    (b3 / "README.md").write_text("Nhà thuốc (DEMO).\n", encoding="utf-8")

    (out / "rulebook.md").write_text(RULEBOOK, encoding="utf-8")
    build_calendar().to_csv(out / "calendar.csv", index=False)

    iso = AS_OF.isoformat()
    Q = []  # (business, question, answer record, solution key / custom variants)
    mock: dict = {}

    def add(biz, q, atype, ans=None, alias=None, variants=None, repair=None):
        Q.append({"id": f"q_demo_{len(Q) + 1:04d}", "business_id": biz, "question": q, "as_of_date": iso})
        rec = {"id": Q[-1]["id"], "type": atype, "answer": ans}
        if alias:
            rec["alias"] = alias
        A.append(rec)
        mock[q] = {"variants": variants, "repair": repair if repair else variants[-1]}

    A: list = []
    ok5 = lambda k: [SOL[k]] * 5
    add("biz_demo1", "sp bán dc nhiều nhất tuần rồi là gì", "string", g1["best_last_week"][0], g1["best_last_week"][1],
        variants=[BROKEN_KEY] + [SOL["best_last_week"]] * 4, repair=SOL["best_last_week"])           # exercises repair
    add("biz_demo1", "tháng này doanh thu bao nhiêu", "number", g1["rev_mtd"], variants=ok5("rev_mtd"))
    add("biz_demo1", "doanh thu tháng 2 so với tháng 1 tăng giảm bao nhiêu phần trăm", "percent", g1["pct_feb_vs_jan"],
        variants=[SOL["pct"], WRONG_PCT, SOL["pct"], SOL["pct"], SOL["pct"]])                        # exercises voting
    add("biz_demo1", "top 3 mặt hàng doanh thu cao nhất tháng trước", "list", g1["top3_rev_feb"], variants=ok5("top3"))
    add("biz_demo1", "tháng trước ngày nào bán được nhiều tiền nhất", "date", g1["best_day_feb"], variants=ok5("best_day"))
    add("biz_demo1", "tuần rồi doanh thu có cao hơn tuần trước đó không", "boolean", g1["lastweek_gt_prevweek"], variants=ok5("week_cmp"))
    add("biz_demo1", "còn ai nợ tiền chưa trả", "table", g1["debt_table"], variants=ok5("debt"))
    add("biz_demo1", "lợi nhuận tháng trước là bao nhiêu", "unanswerable", None, variants=ok5("profit"))
    scattered = [D(f"answer_type = 'number'\nanswer = {g1['rev_yesterday'] + k * 1000}") for k in range(5)]
    add("biz_demo1", "hôm qua cả tiệm thu về bao nhiêu", "number", g1["rev_yesterday"], variants=scattered)  # no agreement -> abstain
    add("biz_demo2", "tuần này bán được bao nhiêu ly", "number", g2["cups_week"], variants=ok5("cups_week"))
    add("biz_demo2", "món nào bán chạy nhất tháng này", "string", g2["best_item_month"][0], g2["best_item_month"][1], variants=ok5("best_item"))
    add("biz_demo3", "kho còn bao nhiêu loại thuốc đã hết hạn", "number", g3["n_expired"], variants=ok5("n_expired"))
    add("biz_demo3", "thuốc nào sắp hết hạn trong 30 ngày tới", "list", g3["soon"], variants=ok5("soon"))

    with open(out / "questions.jsonl", "w", encoding="utf-8") as f:
        for q in Q:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(out / "answers.jsonl", "w", encoding="utf-8") as f:
        for r in A:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "mock_responses.json").write_text(json.dumps(mock, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"demo split written to {out}  ({len(Q)} questions, 3 businesses)")


if __name__ == "__main__":
    main()
