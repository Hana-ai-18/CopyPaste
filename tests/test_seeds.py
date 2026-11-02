"""The hand-written seed examples must run and give the right answer on messy data (they teach the model the helper idioms)."""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datamate.helpers import build_helpers
from datamate.sandbox import patch_pandas_date_compare

ROOT = Path(__file__).resolve().parents[1]


def _orders():
    rows = [
        # order, date (text, mixed formats), status, customer, total, product, qty
        ("D1", "28/03/2026", "Đã hoàn thành", "Nguyễn A", "100,000", "Sữa", 2),
        ("D1", "28/03/2026", "Đã hoàn thành", "Nguyễn A", "100,000", "Bánh", 1),
        ("D2", "28/03/2026", "hủy đơn", "Trần B", "50,000", "Sữa", 1),
        ("D3", "2026-03-28", "xong", "Khách lẻ", "70,000", "Bánh", 5),
        ("D4", "29/03/2026", "completed", "kl", "10,000", "Sữa", 1),
        ("D5", "29/03/2026", "Đã hoàn thành", "Lê C", "300,000", "Sữa", 3),
        ("D5", "29/03/2026", "Đã hoàn thành", "Lê C", "300,000", "Bánh", 1),
        ("D6", "29/03/2026", "trả lại", "Phạm D", "999,000", "Sữa", 9),
        ("D7", "10/06/2025", "Đã hoàn thành", "Nguyễn A", "500,000", "Sữa", 1),
        ("D8", "11/06/2025", "Đã hoàn thành", "khách lẻ", "900,000", "Sữa", 1),
        ("D9", "12/06/2025", "Đã hoàn thành", "Trần B", "200,000", "Bánh", 1),
        ("D10", "05/04/2026", "Đã hoàn thành", "Lê C", "400,000", "Sữa", 1),
        ("D11", "05/05/2026", "Đã hoàn thành", "Lê C", "200,000", "Sữa", 1),
        ("D12", "06/05/2026", "Đã hoàn thành", "Nguyễn A", "190,000", "Sữa", 1),
    ]
    return pd.DataFrame(rows, columns=["Ma don", "Ngay dat", "Trang thai", "Khach hang", "Tong tien", "Ten hang", "So luong"])


def _run(code, as_of):
    patch_pandas_date_compare()
    ns = {"pd": pd, "np": np, "dt": dt, "as_of": as_of, "orders": _orders(), "dfs": {"orders": _orders()}}
    ns.update(build_helpers(as_of, None))
    ns["declare_unanswerable"] = lambda r="": ns.update(answer_type="unanswerable", answer=None)
    exec(compile(code, "<seed>", "exec"), ns)
    return ns["answer_type"], ns.get("answer")


def _seed(prefix):
    for line in open(ROOT / "exemplars" / "seed_exemplars.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d["question"].startswith(prefix):
            return d["code"]
    raise AssertionError(prefix)


def test_seed_file_is_valid():
    items = [json.loads(l) for l in open(ROOT / "exemplars" / "seed_exemplars.jsonl", encoding="utf-8") if l.strip()]
    assert len(items) >= 6 and all({"question", "code", "answer_type"} <= set(i) for i in items)


def test_seed_orders_yesterday():
    assert _run(_seed("hôm qua"), dt.date(2026, 3, 29)) == ("number", 2)          # D1 + D3 (D2 cancelled)


def test_seed_revenue_today_counts_each_order_once():
    assert _run(_seed("hnay"), dt.date(2026, 3, 29)) == ("number", 310000)        # D4 10k + D5 300k once; D6 returned


def test_seed_top_customers_without_anonymous():
    t, a = _run(_seed("top 3"), dt.date(2026, 6, 16))
    assert (t, a) == ("list", ["Nguyễn A", "Trần B"])                            # 'khách lẻ' (900k) is excluded


def test_seed_percent_change_vs_previous_month():
    t, a = _run(_seed("doanh thu tháng 5/2026"), dt.date(2026, 6, 16))
    assert t == "percent" and abs(a - (-2.5)) < 1e-9                               # (390k - 400k) / 400k


def test_seed_unanswerable():
    assert _run(_seed("lợi nhuận"), dt.date(2026, 6, 16))[0] == "unanswerable"


def test_date_compare_patch_is_day_level():
    patch_pandas_date_compare()
    s = pd.Series(pd.to_datetime(["2026-03-28", "2026-03-29 10:00:00", "2026-03-30"], format="mixed"))
    d = dt.date(2026, 3, 29)
    assert (s == d).tolist() == [False, True, False]
    assert (s >= d).tolist() == [False, True, True] and (s <= d).tolist() == [True, True, False]
    assert (s > d).tolist() == [False, False, True] and (s < d).tolist() == [True, False, False]


def test_anon_customer_variants():
    from datamate.helpers import anon_customer
    s = pd.Series(["Khách lẻ", "khách lẻ", "kl", "lẻ", "KL", "Khách vãng lai", "Khách (không lưu)", "", None, float("nan"),
                   "Thân Cường Bình", "Lê", "Lê Văn A", "khách sỉ", "Nguyễn Lẻ"])
    got = anon_customer(s).tolist()
    assert got == [True] * 10 + [False] * 5
