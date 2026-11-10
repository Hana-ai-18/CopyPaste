import importlib

import pandas as pd

from datamate.intent import audit_code


def test_lost_columns_detects_line_amounts():
    from datamate.lineage import lost_columns

    df = pd.DataFrame({"oid": ["A", "A", "A", "B", "B"], "tien": ["1,000", "2,000", "3,000", "4,000", "5,000"], "kh": ["x", "x", "x", "y", "y"], "ngay": ["d1"] * 5})
    assert lost_columns(df, "oid") == ["tien"]                 # kh / ngay are constant inside an order, text is not numeric-like


def test_install_records_only_lossy_dedupe():
    import datamate.lineage as L

    orig = pd.DataFrame.drop_duplicates
    sink: list = []
    try:
        L.install(sink)
        df = pd.DataFrame({"oid": ["A", "A", "B"], "tien": [1, 2, 3], "kh": ["x", "x", "y"]})
        out = df.drop_duplicates("oid")
        assert len(out) == 2 and sink and sink[0]["lost"] == ["tien"]
        sink.clear()
        df[["oid", "kh"]].drop_duplicates("oid")                # nothing numeric is lost
        df.drop_duplicates()                                    # no subset: exact duplicates only
        assert sink == []
    finally:
        pd.DataFrame.drop_duplicates = orig
        importlib.reload(L)


def test_audit_uses_lineage_not_names():
    ev = [{"key": ["oid"], "lost": ["so_tien_x"], "rows_before": 10, "rows_after": 4}]
    assert audit_code("tong theo don", "per = t.drop_duplicates('oid')\nanswer = per['so_tien_x'].sum()", "number", 5, None, ev)
    assert audit_code("tong theo don", "n = t.drop_duplicates('oid').shape[0]\nanswer = n", "number", 5, None, ev) == []   # does not use the lost column


def test_unparsed_and_unclassified_and_fanout():
    import importlib

    import datamate.lineage as L
    from datamate.helpers import norm_status, parse_dates, to_num

    ns = {"parse_dates": parse_dates, "to_num": to_num, "norm_status": norm_status}
    sink: list = []
    L.wrap_helpers(ns, sink)
    d = pd.Series(["01/02/2025"] * 20 + ["not a date"] * 5, name="Ngày")
    ns["parse_dates"](d)
    ns["to_num"](pd.Series(["10"] * 20 + ["abc", "def", "xyz"], name="Tien"))
    ns["norm_status"](pd.Series(["xong"] * 50 + ["quá lạ", "quá lạ", "quá lạ", "lạ lạ"], name="TT"))
    kinds = [e["kind"] for e in sink]
    assert kinds.count("unparsed") == 2 and "unclassified_status" in kinds
    orig = pd.DataFrame.merge
    try:
        L.install_merge(sink)
        a = pd.DataFrame({"k": [1, 2], "tong": [100, 200]})
        b = pd.DataFrame({"k": [1, 1, 2, 2], "x": [1, 2, 3, 4]})
        a.merge(b, on="k")
        assert sink[-1]["kind"] == "fanout" and "tong" in sink[-1]["left_numeric"]
    finally:
        pd.DataFrame.merge = orig
        importlib.reload(L)
    msgs = audit_code("tong", "m = a.merge(b, on='k')\nanswer = m['tong'].sum()", "number", 1, None, [sink[-1]])
    assert msgs and "multiplied" in msgs[0]
    assert audit_code("tong", "x=1", "number", 1, None, [sink[0]])
