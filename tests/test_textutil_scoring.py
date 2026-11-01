import json

import numpy as np
import pandas as pd
import pytest

from datamate.answer_format import answer_key, round_half_up, sanitize_answer, to_py
from datamate.scoring import penalized_accuracy, score_one
from datamate.textutil import norm_string, strip_accents, vi_key


# ----------------------------------------------------------------------------- text
def test_norm_string_official_rules():
    assert norm_string("Sting dâu 330ml") == "sting dâu 330ml"
    assert norm_string("  Cà-phê   SỮA!! ") == "cà phê sữa"
    assert norm_string("Xi măng Hà Tiên PCB40 (bao 50kg)") == "xi măng hà tiên pcb40 bao 50kg"
    # NFC vs NFD input are the same
    import unicodedata

    assert norm_string(unicodedata.normalize("NFD", "Trà đào")) == norm_string("Trà đào") == "trà đào"
    # diacritics are NOT removed by the official normalisation
    assert norm_string("Trà") != norm_string("Tra")


def test_accent_helpers():
    assert strip_accents("Đường Biên Hòa") == "Duong Bien Hoa"
    assert vi_key("Bạc XỈU!") == "bac xiu"


# ----------------------------------------------------------------------------- rounding
def test_round_half_up():
    assert round_half_up(2.5) == 3 and round_half_up(-2.5) == -3 and round_half_up(0.5) == 1
    assert round_half_up(12.345, 2) == 12.35 and round_half_up(-12.445, 2) == -12.45
    assert round_half_up(np.float64(1.005), 2) == 1.01  # repr-based, not binary float based


# ----------------------------------------------------------------------------- per-question scoring
def S(pred, ref):
    return score_one(pred, ref)[0]


def test_number_rules():
    ref = {"type": "number", "answer": 48250000}
    assert S({"type": "number", "answer": 48250000}, ref) == 1
    assert S({"type": "number", "answer": 48250000.0}, ref) == 1
    assert S({"type": "number", "answer": 48250001}, ref) == -1
    assert S({"type": "number", "answer": "48250000"}, ref) == -1         # a string is not a number
    refd = {"type": "number", "answer": 3.46}
    assert S({"type": "number", "answer": 3.456}, refd) == 1               # rounded to 2 decimals
    assert S({"type": "number", "answer": 3.4}, refd) == -1


def test_percent_sign_matters():
    ref = {"type": "percent", "answer": -12.45}
    assert S({"type": "percent", "answer": -12.45}, ref) == 1
    assert S({"type": "percent", "answer": 12.45}, ref) == -1
    assert S({"type": "number", "answer": -12.45}, ref) == -1             # wrong type = wrong


def test_string_normalisation_and_alias():
    ref = {"type": "string", "answer": "Xi măng Hà Tiên PCB40 bao 50kg", "alias": ["SP0001"]}
    assert S({"type": "string", "answer": "xi măng hà tiên pcb40, bao 50kg"}, ref) == 1
    assert S({"type": "string", "answer": "SP0001"}, ref) == 1
    assert S({"type": "string", "answer": "Xi mang Ha Tien PCB40 bao 50kg"}, ref) == -1   # accents matter


def test_date_boolean():
    assert S({"type": "date", "answer": "2026-08-15"}, {"type": "date", "answer": "2026-08-15"}) == 1
    assert S({"type": "date", "answer": "2026-08"}, {"type": "date", "answer": "2026-08-15"}) == -1
    assert S({"type": "boolean", "answer": True}, {"type": "boolean", "answer": True}) == 1
    assert S({"type": "boolean", "answer": False}, {"type": "boolean", "answer": True}) == -1


def test_list_is_ordered():
    ref = {"type": "list", "answer": ["Cà phê sữa", "Bạc xỉu", "Trà đào"]}
    assert S({"type": "list", "answer": ["cà phê sữa", "bạc xỉu", "trà đào"]}, ref) == 1
    assert S({"type": "list", "answer": ["Bạc xỉu", "Cà phê sữa", "Trà đào"]}, ref) == -1
    assert S({"type": "list", "answer": ["Cà phê sữa", "Bạc xỉu"]}, ref) == -1


def test_table_columns_by_name_rows_as_set():
    ref = {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["Chị Lan", 3200000], ["Anh Tú", 1500000]]}}
    same = {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["Anh Tú", 1500000], ["Chị Lan", 3200000]]}}
    swapped = {"type": "table", "answer": {"columns": ["so_tien_no", "khach_hang"], "rows": [[1500000, "Anh Tú"], [3200000, "Chị Lan"]]}}
    renamed = {"type": "table", "answer": {"columns": ["ten", "no"], "rows": [["Chị Lan", 3200000], ["Anh Tú", 1500000]]}}   # by position
    wrong = {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["Chị Lan", 3200000], ["Anh Tú", 1500001]]}}
    assert S(same, ref) == 1 and S(swapped, ref) == 1 and S(renamed, ref) == 1
    assert S(wrong, ref) == -1


def test_abstain_and_unanswerable():
    num = {"type": "number", "answer": 5}
    un = {"type": "unanswerable", "answer": None}
    assert score_one({"type": "abstain"}, num) == (0, "abstain")
    assert score_one(None, num) == (0, "abstain")
    assert S({"type": "unanswerable"}, un) == 1
    assert S({"type": "number", "answer": 1}, un) == -1                   # any answer to an unanswerable question is wrong
    assert S({"type": "unanswerable"}, num) == -1                        # claiming unanswerable on an answerable one is wrong
    assert score_one({"type": "abstain"}, un) == (0, "abstain")


def test_penalized_accuracy_formula():
    refs = [{"id": "a", "type": "number", "answer": 1}, {"id": "b", "type": "number", "answer": 2},
            {"id": "c", "type": "number", "answer": 3}, {"id": "d", "type": "unanswerable", "answer": None}]
    preds = {"a": {"id": "a", "type": "number", "answer": 1},   # +1
             "b": {"id": "b", "type": "number", "answer": 9},   # -1
             "c": {"id": "c", "type": "abstain"}}               # 0 ; d missing -> 0
    m = penalized_accuracy(preds, refs)
    assert m["PA"] == pytest.approx(0.0) and (m["correct"], m["wrong"], m["abstain"]) == (1, 1, 2)
    preds["d"] = {"id": "d", "type": "unanswerable"}
    assert penalized_accuracy(preds, refs)["PA"] == pytest.approx(0.25)


# ----------------------------------------------------------------------------- answer formatting
def test_sanitize_numpy_and_pandas_values():
    assert sanitize_answer("number", np.int64(5)) == 5 and isinstance(sanitize_answer("number", np.int64(5)), int)
    assert sanitize_answer("number", 5.0) == 5
    assert sanitize_answer("number", 3.456) == 3.46
    assert sanitize_answer("number", pd.Series([7])) == 7
    assert sanitize_answer("percent", -12.445) == -12.45
    assert sanitize_answer("date", pd.Timestamp("2026-03-04 10:00")) == "2026-03-04"
    assert sanitize_answer("date", "04/03/2026") == "2026-03-04"
    assert sanitize_answer("date", "2026-03") == "2026-03"
    assert sanitize_answer("boolean", np.bool_(True)) is True
    assert sanitize_answer("list", pd.Series(["a", "b"])) == ["a", "b"]
    assert sanitize_answer("list", [np.int64(1), 2.5]) == [1, 2.5]
    df = pd.DataFrame({"ten": ["A", "B"], "so": [np.int64(1), np.int64(2)]})
    t = sanitize_answer("table", df)
    assert t == {"columns": ["ten", "so"], "rows": [["A", 1], ["B", 2]]}
    json.dumps(t)  # JSON-able


@pytest.mark.parametrize("atype,val", [("number", float("nan")), ("number", True), ("number", "12"), ("string", "  "), ("date", "hôm qua"),
                                       ("boolean", "yes"), ("list", [[1]]), ("table", {"columns": ["a"], "rows": [[1, 2]]}), ("weird", 1), ("number", None)])
def test_sanitize_rejects_bad_values(atype, val):
    with pytest.raises(ValueError):
        sanitize_answer(atype, val)


def test_answer_key_for_voting():
    assert answer_key("string", "Trà  đào!") == answer_key("string", "trà đào")
    assert answer_key("number", 5) == answer_key("number", 5.0)
    assert answer_key("list", ["a", "b"]) != answer_key("list", ["b", "a"])           # order matters for lists
    t1 = {"columns": ["A", "b"], "rows": [["x", 1], ["y", 2]]}
    t2 = {"columns": ["a", "B"], "rows": [["y", 2], ["x", 1]]}
    assert answer_key("table", t1) == answer_key("table", t2)                         # row order does not
    assert to_py(pd.NaT) is None
