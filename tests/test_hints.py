import pandas as pd

from datamate.agent import empty_reason
from datamate.hints import structure_hints
from datamate.profile import profile_tables


def _df():
    rows = []
    for i in range(60):
        for j in range(3):
            rows.append({"Ma don": f"D{i}", "Trang thai": ["xong", "huy don", "Da hoan thanh"][i % 3], "Doi tac": "Khach le" if i % 2 else f"KH{i}",
                         "San pham": f"SP{j}", "Thanh tien": 1000 * (j + 1)})
    return pd.DataFrame(rows)


def test_structure_hints_levels_status_customer():
    h = structure_hints({"t": _df()})
    assert "order id column 'Ma don'" in h
    order_part = h.split("LINE-level")[0]
    assert "Trang thai" in order_part and "Doi tac" in order_part
    assert "Thanh tien" in h.split("LINE-level")[1]
    assert "status column 'Trang thai'" in h and "->huy" in h
    assert "anonymous" in h and "Khach le" in h


def test_hints_first_in_profile_and_survive_truncation():
    txt = profile_tables({"t": _df()}, max_chars=900)
    assert txt.startswith("## Detected structure")
    assert len(txt) <= 900 + 60


def test_hints_never_crash_on_odd_tables():
    assert isinstance(structure_hints({"e": pd.DataFrame(), "x": pd.DataFrame({"a": [1, 2]})}), str)


def test_empty_reason():
    assert empty_reason("number", 0)
    assert empty_reason("list", [])
    assert empty_reason("table", {"columns": ["a"], "rows": []})
    assert empty_reason("number", 5) is None
    assert empty_reason("boolean", False) is None


def test_capability_hints_detect_present_and_missing():
    from datamate.hints import capability_hints

    sales = pd.DataFrame({"MaHoaDon": ["a", "b"], "TongTien": [1, 2], "KhachDaTra": [1, 2], "ConPhaiThu": [0, 0]})
    stock = pd.DataFrame({"MaHang": ["x"], "TonCuoi": [3]})
    caps = "\n".join(capability_hints({"don_hang": sales, "the_kho": stock}))
    assert "debt" in caps.split("NOT found")[0] and "stock levels" in caps.split("NOT found")[0]
    assert "NOT found" in caps and "expiry" in caps.split("NOT found")[1]
    none = "\n".join(capability_hints({"x": pd.DataFrame({"a": [1], "dinh_muc_ton_toi_thieu": [1]})}))
    assert "stock levels" in none.split("NOT found")[1]          # min/max stock thresholds are NOT stock levels


def test_wide_schema_keeps_every_table():
    tabs = {f"t{i}": pd.DataFrame({f"column_number_{j}": ["some text value"] * 3 for j in range(14)}) for i in range(14)}
    txt = profile_tables(tabs, max_chars=2500)
    assert all(f"t{i}" in txt for i in range(14))
