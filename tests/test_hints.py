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


def _fake_rulebook(rows_per=40):
    gen = "".join(f"| Luật chung {i} về doanh thu và đơn hàng số {i} | nội dung dài dài dài dài dài dài dài dài dài |\n" for i in range(rows_per))
    ind = "".join(f"### {n}\n| Luật | Mặc định |\n|---|---|\n| luật ngành {n} | nội dung |\n\n" for n in ["Cà phê", "Nhà thuốc", "Sửa xe"])
    return ("# R\n\n## 1. Đơn hàng\n| Luật | Mặc định |\n|---|---|\n" + gen + "\n## 6. Theo ngành\nchung\n\n" + ind +
            "## 7. Không trả lời được\nKHÔNG-TRẢ-LỜI-MARKER\n\n## 8. Cách ghi trong file\nCÁCH-GHI-MARKER\n\n## 9. Đáp án và chấm điểm\nFORMAT-MARKER\n")


def test_rulebook_never_cut_in_the_middle_and_keeps_format_chapters():
    from datamate.rulebook import select_rulebook

    rb = _fake_rulebook()
    out = select_rulebook(rb, "File Excel", 2500, evidence="doanh thu don hang so 7")
    assert len(out) <= 2500 and "\n...\n" not in out
    assert all(m in out for m in ("KHÔNG-TRẢ-LỜI-MARKER", "CÁCH-GHI-MARKER", "FORMAT-MARKER"))
    assert "Luật chung 7 " in out                       # row that matches the shop's own questions survives pruning


def test_rulebook_industry_chosen_from_evidence_not_readme():
    from datamate.rulebook import select_rulebook

    rb = _fake_rulebook(5)
    out = select_rulebook(rb, "File Excel xuất từ phần mềm", len(rb) - 50, evidence="phieu bao hanh sua xe luot xe bien so sua xe")
    assert "luật ngành Sửa xe" in out and "luật ngành Cà phê" not in out
    keep_all = select_rulebook(rb, "File Excel", len(rb) - 50, evidence="chung chung")
    assert all(n in keep_all for n in ["Cà phê", "Nhà thuốc", "Sửa xe"])        # no evidence -> never guess


def test_variant_hint_groups_same_person_spellings():
    from datamate.hints import _variant_hint

    df = pd.DataFrame({"Thợ": ["A.Đức", "Đức", "a Đức", "a Phúc", "A.Phúc", "Phúc", "An", "a An"] * 3, "Tiền": range(24)})
    h = "\n".join(_variant_hint(df))
    assert "'Thợ'" in h and "3 distinct" in h.replace("only 3 distinct", "3 distinct")
    plain = pd.DataFrame({"x": ["Trà đào", "Cà phê", "Bạc xỉu", "Sting"] * 5})
    assert _variant_hint(plain) == []


def test_profile_flags_month_word_and_mixed_dates():
    s = pd.DataFrame({"Ngày tháng": ["1 thg 6 2025", "2 thg 6 2025", "3 thg 6 2025", "4 thg 6 2025", "5 thg 7 2025"] * 3, "x": range(15)})
    assert "DATE-LIKE" in profile_tables({"t": s})
    m = pd.DataFrame({"Ngày": ["2025-01-01 00:00:00", "01/01/2025", "02/01/2025", "03/01/2025", "2025-01-05"] * 3, "x": range(15)})
    assert "DATE-LIKE" in profile_tables({"t": m})
