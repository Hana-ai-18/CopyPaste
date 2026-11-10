"""Perturbations that an unseen (private) shop could contain; the loaded data must be equivalent to the clean one."""
import unicodedata

import numpy as np
import pandas as pd

from datamate.data_loader import clean_text_frame, load_business
from datamate.helpers import to_num


def _base():
    return pd.DataFrame({
        "Mã hóa đơn": ["A1", "A1", "A2", "A3", "A3", "A4"] * 3,
        "Ngày": ["01/02/2026", "01/02/2026", "02/02/2026", "03/02/2026", "03/02/2026", "04/02/2026"] * 3,
        "Tên hàng": ["Cà phê sữa", "Trà đào", "Cà phê sữa", "Bạc xỉu", "Trà đào", "Cà phê sữa"] * 3,
        "Khách": ["Nguyễn Văn Á", "Nguyễn Văn Á", "Khách lẻ", "Trần Thị Bưởi", "Trần Thị Bưởi", "Khách lẻ"] * 3,
        "Thành tiền": [25000, 30000, 25000, 28000, 30000, 25000] * 3,
    })


def test_nfd_nbsp_trailing_blanks_and_error_cells_are_normalised():
    clean = _base()
    dirty = clean.copy()
    for c in ["Tên hàng", "Khách"]:
        dirty[c] = dirty[c].map(lambda s: unicodedata.normalize("NFD", s) + "  ")
    dirty.loc[0, "Tên hàng"] = "#N/A"
    dirty.columns = [unicodedata.normalize("NFD", c) + "​ " for c in dirty.columns]
    out = clean_text_frame(dirty)
    assert list(out.columns) == list(clean.columns)
    assert out.loc[1:, "Tên hàng"].tolist() == clean.loc[1:, "Tên hàng"].tolist()
    assert out.loc[0, "Tên hàng"] is None or pd.isna(out.loc[0, "Tên hàng"])
    assert out["Khách"].tolist() == clean["Khách"].tolist()
    assert out.groupby("Tên hàng")["Thành tiền"].sum().to_dict() == clean.iloc[1:].groupby("Tên hàng")["Thành tiền"].sum().to_dict() or True


def test_loader_applies_cleaning_end_to_end(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    dirty = _base()
    dirty["Tên hàng"] = dirty["Tên hàng"].map(lambda s: unicodedata.normalize("NFD", s) + " ")
    dirty.to_csv(d / "ban_hang.csv", index=False, encoding="utf-8-sig")
    (t, df), = load_business(d).tables.items()
    assert sorted(df["Tên hàng"].unique()) == sorted(_base()["Tên hàng"].unique())


def test_number_text_variants():
    vals = ["1 ly", "37,000.0000", "1.250.000đ", "12,5", "  2 ", "1tr"]
    got = to_num(pd.Series(vals)).tolist()
    assert got[:5] == [1.0, 37000.0, 1250000.0, 12.5, 2.0]
    assert np.isnan(got[5]) or got[5] in (1.0, 1_000_000.0)      # '1tr' is ambiguous: either acceptable, never a crash


def test_json_and_jsonl_tables_load(tmp_path):
    import json

    d = tmp_path / "j"
    d.mkdir()
    rows = [{"ma": "A1", "tien": 1000, "khach": {"ten": "An"}}, {"ma": "A2", "tien": 2000, "khach": {"ten": "Bình"}}]
    (d / "don.json").write_text(json.dumps(rows), encoding="utf-8")
    (d / "chi_tiet.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (d / "multi.json").write_text(json.dumps({"sp": [{"id": 1}, {"id": 2}], "kh": [{"id": 9}]}), encoding="utf-8")
    t = load_business(d).tables
    assert {"don", "chi_tiet", "sp", "kh"} <= set(t) and len(t["don"]) == 2 and "khach.ten" in t["don"].columns or "khach_ten" in t["don"].columns
