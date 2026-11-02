import datetime as dt

import pandas as pd

from datamate import refcalc as R
from datamate.helpers import to_num


def _shop(star=True):
    rows, lines = [], []
    k = 0
    for day in pd.date_range("2025-01-01", "2025-03-31"):
        for j in range(3):
            k += 1
            st = "cancelled" if k % 10 == 0 else "completed"
            rows.append({"Mã chứng từ": f"ORD{k:06d}", "Ngày": day.strftime("%Y-%m-%d"), "Tình trạng": st})
            lines.append({"Mã đơn hàng": f"ORD{k:06d}", "Amount": f"{(j + 1) * 10}.000đ", "SL": 1})
            lines.append({"Mã đơn hàng": f"ORD{k:06d}", "Amount": "1tr090", "SL": 1})
    return {"don_hang": pd.DataFrame(rows), "chi_tiet_don": pd.DataFrame(lines)}


def _truth(dfs, a, b):
    h, d = dfs["don_hang"], dfs["chi_tiet_don"]
    d = d.assign(a=to_num(d["Amount"])).groupby("Mã đơn hàng")["a"].sum()
    h = h.assign(a=h["Mã chứng từ"].map(d), dd=pd.to_datetime(h["Ngày"]))
    w = h[(h.dd >= a) & (h.dd <= b)]
    ok = w[w["Tình trạng"] != "cancelled"]
    return ok["a"].sum(), len(ok), int((w["Tình trạng"] == "cancelled").sum()), len(w)


def test_to_num_tr_forms():
    s = to_num(pd.Series(["1tr090", "9tr", "2tr5", "1k5", "50k", "39.000đ"]))
    assert s.tolist() == [1090000, 9000000, 2500000, 1500, 50000, 39000]


def test_star_schema_induced():
    spec = R.induce(_shop())
    assert spec and spec["detail"] == "chi_tiet_don" and spec["oid"] == "Mã chứng từ"


def test_families_match_ground_truth():
    dfs = _shop()
    asof = dt.date(2025, 5, 10)
    rev, cnt, canc, tot = _truth(dfs, "2025-02-01", "2025-02-28")
    assert R.compute(dfs, "Doanh thu tháng 2/2025 là bao nhiêu?", asof) == ("number", int(round(rev)))
    assert R.compute(dfs, "Trung bình mỗi đơn tháng 2/2025 mang về bao nhiêu dthu", asof) == ("number", int(round(rev / cnt)))
    r = R.compute(dfs, "Tháng 2/2025 cửa hàng có bao nhiêu % đơn bị huỷ?", asof)
    assert r == ("percent", round(100 * canc / tot, 2))
    rev1, *_ = _truth(dfs, "2025-01-01", "2025-01-31")
    r = R.compute(dfs, "So với tháng liền trước, doanh thu tháng 2/2025 thay đổi bao nhiêu %?", asof)
    assert r == ("percent", round(100 * (rev - rev1) / rev1, 2))


def test_no_opinion_outside_scope():
    dfs = _shop()
    asof = dt.date(2025, 5, 10)
    assert R.compute(dfs, "Doanh thu của Trà đào tháng 2/2025?", asof) is None
    assert R.compute(dfs, "Doanh thu tháng 2/2020 là bao nhiêu?", asof) is None


def test_year_less_dates_are_not_order_dates():
    df = pd.DataFrame({"Ngày lập": ["1/1", "2/1"] * 20, "Giờ": ["2025-01-01 06:00:00", "2025-01-02 07:00:00"] * 20})
    assert not R._is_dateish(df, "Ngày lập") and R._is_dateish(df, "Giờ")
