import datetime as dt

import pandas as pd
import pytest

from datamate import helpers as H
from datamate.data_loader import load_business, read_csv_robust, safe_identifier
from datamate.profile import profile_tables

D = dt.date


# ----------------------------------------------------------------------------- date helpers
def test_week_and_month_defaults():
    asof = D(2026, 3, 4)  # Wednesday
    assert H.week_range(asof) == (D(2026, 3, 2), D(2026, 3, 8))
    assert H.last_week(asof) == (D(2026, 2, 23), D(2026, 3, 1))
    assert H.this_week_to_date(asof) == (D(2026, 3, 2), asof)
    assert H.this_month_to_date(asof) == (D(2026, 3, 1), asof)
    assert H.last_month(asof) == (D(2026, 2, 1), D(2026, 2, 28))
    assert H.last_month(D(2026, 1, 15)) == (D(2025, 12, 1), D(2025, 12, 31))
    assert H.same_period_prev_month(D(2026, 3, 31)) == (D(2026, 2, 1), D(2026, 2, 28))
    assert H.last_quarter(asof) == (D(2025, 10, 1), D(2025, 12, 31))
    assert H.last_quarter(D(2026, 8, 1)) == (D(2026, 4, 1), D(2026, 6, 30))
    assert H.last_n_days(asof, 7) == (D(2026, 2, 25), D(2026, 3, 3))
    assert H.last_n_days(asof, 7, include_today=True) == (D(2026, 2, 26), asof)
    assert H.same_period_last_year(D(2024, 2, 29), D(2024, 3, 5)) == (D(2023, 2, 28), D(2023, 3, 5))
    assert H.week_range(D(2026, 3, 8)) == (D(2026, 3, 2), D(2026, 3, 8))  # Sunday belongs to the Monday-start week


def test_rounding_and_pct():
    assert H.round_vnd(1234.5) == 1235 and H.round_vnd(-0.5) == -1
    assert H.round2(2.675) == 2.68
    assert H.pct_change(87.55, 100) == -12.45
    with pytest.raises(ValueError):
        H.pct_change(5, 0)


def test_parse_dates_and_between():
    s = pd.Series(["2026-03-01", "04/03/2026", "2026-03-02 10:30:00", None])
    p = H.parse_dates(s)
    assert p.iloc[1] == pd.Timestamp("2026-03-04") and p.iloc[2] == pd.Timestamp("2026-03-02 10:30")
    assert p.iloc[0] == pd.Timestamp("2026-03-01") and pd.isna(p.iloc[3])
    dmy = pd.Series(["31/12/2025", "01/01/2026", "05/03/2026 14:30"])
    assert list(H.parse_dates(dmy)) == [pd.Timestamp("2025-12-31"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-05 14:30")]
    ymd = pd.Series(["2026/3/4", "2026.03.05", "2026-03-06"])                  # year-first with other separators
    assert list(H.parse_dates(ymd)) == [pd.Timestamp("2026-03-04"), pd.Timestamp("2026-03-05"), pd.Timestamp("2026-03-06")]
    # ISO values must never be read day-first even when the column also holds d/m/Y values
    mixed = pd.Series(["2026-03-02", "13/03/2026", "2026-02-05"])
    assert list(H.parse_dates(mixed)) == [pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-13"), pd.Timestamp("2026-02-05")]
    iso = pd.Series(["2026-03-01 23:59:59", "2026-03-04", "2026-03-05"])
    assert list(H.between(iso, D(2026, 3, 1), D(2026, 3, 4))) == [True, True, False]   # inclusive, time part ignored
    excel = pd.Series([46082, 46083])                                                  # Excel serial dates
    assert H.parse_dates(excel).iloc[0].year == 2026


def test_rank_top_tie_break_by_code():
    df = pd.DataFrame({"code": ["B", "A", "C", "D"], "v": [10, 10, 7, 12]})
    r = H.rank_top(df, "v", "code", 3)
    assert list(r.code) == ["D", "A", "B"]                      # 12, then the tie 10 -> code ascending
    assert list(H.rank_top(df, "v", "code", 2, ascending=True).code) == ["C", "A"]


def test_fuzzy_and_contains_vi():
    names = ["Trà đào cam sả", "Cà phê sữa", "Bạc xỉu"]
    assert H.fuzzy_find(names, "tra dao")[0][0] == "Trà đào cam sả"
    assert H.contains_vi(pd.Series(names), "CA PHE").tolist() == [False, True, False]


# ----------------------------------------------------------------------------- lunar calendar
def test_tet_window_from_calendar(demo_split):
    cal = pd.read_csv(demo_split / "calendar.csv")
    lun = H.LunarCalendar(cal)
    assert lun.tet_date(2026) == D(2026, 2, 17) and lun.tet_date(2025) == D(2025, 1, 29)
    start, end = lun.tet_window(2026)
    assert lun.lunar_of(start) == (12, 23, False) and lun.lunar_of(end) == (1, 10, False)
    assert end == D(2026, 2, 26) and start < D(2026, 2, 17)
    fns = H.build_helpers(D(2026, 3, 4), lun)
    assert fns["tet_window"](2026) == (start, end)
    with pytest.raises(ValueError):
        lun.tet_date(1999)


def test_helper_docs_listing():
    txt = H.helper_docs()
    assert "last_week(" in txt and "rank_top(" in txt and "tet_window(" in txt


# ----------------------------------------------------------------------------- loader
def test_load_demo_businesses(demo_split):
    b1 = load_business(demo_split / "data" / "biz_demo1")
    assert set(b1.tables) == {"products", "customers", "orders", "order_items"} and "tạp hóa" in b1.readme
    assert str(b1.tables["orders"]["order_date"].dtype) in ("object", "str", "string")

    b2 = load_business(demo_split / "data" / "biz_demo2")          # xlsx with 2 junk title rows above the header
    assert set(b2.tables) == {"menu", "sales"}
    sales = b2.tables["sales"]
    assert list(sales.columns) == ["sale_date", "item_code", "cups", "amount"]
    assert pd.api.types.is_datetime64_any_dtype(sales["sale_date"]) and sales["cups"].dtype.kind == "i"

    b3 = load_business(demo_split / "data" / "biz_demo3")          # sqlite
    assert set(b3.tables) == {"medicines", "batches"} and len(b3.tables["batches"]) == 26


def test_csv_semicolon_and_encoding(tmp_path):
    p = tmp_path / "x.csv"
    p.write_bytes("ten;gia\nTrà đào;15000\nCà phê;20000\n".encode("cp1258"))
    df = read_csv_robust(p)
    assert list(df.columns) == ["ten", "gia"] and df["ten"].iloc[0] == "Trà đào" and df["gia"].sum() == 35000
    p2 = tmp_path / "y.csv"
    p2.write_bytes("﻿id,ten\n1,Sữa\n".encode("utf-8"))
    assert list(read_csv_robust(p2).columns) == ["id", "ten"]       # BOM stripped


def test_duplicate_columns_and_identifiers(tmp_path):
    (tmp_path / "Đơn Hàng 2025.csv").write_text("a,a,b\n1,2,3\n", encoding="utf-8")
    b = load_business(tmp_path)
    assert list(b.tables) == ["don_hang_2025"] and list(b.tables["don_hang_2025"].columns) == ["a", "a.1", "b"]
    assert safe_identifier("2025 sales") == "t_2025_sales" and safe_identifier("!!!") == "table"


def test_broken_file_does_not_kill_business(tmp_path):
    (tmp_path / "ok.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "bad.xlsx").write_bytes(b"not an excel file")
    b = load_business(tmp_path)
    assert "ok" in b.tables and any("bad.xlsx" in w for w in b.warnings)


# ----------------------------------------------------------------------------- profile
def test_profile_content_and_budget(demo_split):
    b = load_business(demo_split / "data" / "biz_demo1")
    txt = profile_tables(b.tables, max_chars=7000)
    assert "completed" in txt and "cancelled" in txt              # categorical values are listed
    assert "DATE-LIKE" in txt and "parse_dates" in txt             # text dates are flagged
    assert "appears in" in txt                                     # join hints
    short = profile_tables(b.tables, max_chars=1500)
    assert len(short) <= 1500 and "order_items" in short
