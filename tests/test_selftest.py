"""Chạy: pytest -q  (đặt DATAMATE_TRAIN=/đường/dẫn/train để chạy thêm kiểm tra trên dữ liệu train)."""
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _run(args):
    return subprocess.run([sys.executable, str(ROOT / "datamate_pipeline.py"), *args],
                          capture_output=True, text=True, timeout=1800)


def test_selftest_no_data():
    r = _run(["selftest"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SELFTEST: OK" in r.stdout


def test_selftest_train():
    train = os.environ.get("DATAMATE_TRAIN")
    if not train:
        import pytest
        pytest.skip("chưa đặt DATAMATE_TRAIN")
    r = _run(["selftest", "--split_dir", train])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SELFTEST: OK" in r.stdout


def test_vn_normalize_elongation():
    from vn_normalize import Normalizer
    n = Normalizer()
    for raw, want in [("coasaaaa", "có"), ("nhaatstttt", "nhất"), ("tieeenffff", "tiền")]:
        assert n.normalize(raw)["text"] == want, raw


def test_context_normalizer():
    from vn_normalize import Normalizer
    from vn_context import ContextNormalizer
    texts = ["Tên khách hàng mua nhiều nhất tháng này là gì?", "Sản phẩm nào bán chạy nhất tháng trước?",
             "Doanh thu tháng trước là bao nhiêu?", "Khách hàng nào nợ nhiều nhất?"]
    cn = ContextNormalizer(Normalizer([], [], texts), texts)
    for raw, want in [("teenn khach hang mua nhieuf nhaats", "tên khách hàng mua nhiều nhất"),
                      ("sp nafo baan chayj nhaatstt thangs truoc", "sản phẩm nào bán chạy nhất tháng trước"),
                      ("ddoanhh thuuu thang truoc bnhiuuu", "doanh thu tháng trước bao nhiêu")]:
        assert cn.normalize(raw)["text"] == want, (raw, cn.normalize(raw)["text"])


def test_lexicon_files():
    import yaml
    from vn_normalize import N_TEEN_FILE
    assert N_TEEN_FILE >= 200, "lexicon/teencode.tsv chưa nạp"
    cat = yaml.safe_load(open(ROOT / "lexicon" / "error_catalog.yaml", encoding="utf-8"))
    ids = [e["id"] for grp in cat.values() for e in grp]
    assert len(ids) == len(set(ids)), "trùng id trong error_catalog.yaml"
    assert all(e.get("trang_thai") in ("da_code", "thiet_ke") for grp in cat.values() for e in grp)


def test_ood_spelling():
    from vn_normalize import Normalizer, vni_decode, teen_variants, is_syllable
    from vn_context import ContextNormalizer
    assert vni_decode("tie6n2") == "tiền" and vni_decode("nguo72i") == "người"
    assert vni_decode("SP0066") is None and vni_decode("q1") is None and vni_decode("t7") is None
    assert "thích" in teen_variants("thík") and "phải" in teen_variants("fải")
    assert not is_syllable("kùng") and is_syllable("kiểm") and not is_syllable("ge")
    texts = ["Doanh thu tháng này nhiều hơn tháng trước không?", "Sản phẩm nào bán chạy nhất tuần này vậy?",
             "Cửa hàng còn tồn kho bao nhiêu cái?", "Khách nào nợ nhiều nhất là ai?"]
    cn = ContextNormalizer(Normalizer([], [], texts), texts)
    for raw, want in [("dthu thág nì nhìu hơng thág trc ko", "doanh thu tháng này nhiều hơn tháng trước không"),
                      ("tie6n2 kha1ch no75 nhie62u nha6t1", "tiền khách nợ nhiều nhất")]:
        assert cn.normalize(raw)["text"] == want, (raw, cn.normalize(raw)["text"])
    # câu sạch có dấu không bị đụng
    for t in texts:
        assert cn.normalize(t)["text"].lower().rstrip("?") == t.lower().rstrip("?")


def test_llm_choice_plumbing():
    from vn_normalize import Normalizer
    from vn_context import ContextNormalizer, resolve_unsure, parse_choice
    texts = ["Số bill hôm nay là bao nhiêu?", "So với tháng trước thì sao?"]
    cn = ContextNormalizer(Normalizer([], ["bill"], texts), texts)
    raws = ["so bill hnay", "so voi thang truoc"]
    first = [cn.normalize(r) for r in raws]
    same, _ = resolve_unsure(cn, raws, lambda msgs: ["A"] * len(msgs))       # A = lựa chọn hiện tại -> không đổi
    assert [r["text"] for r in same] == [r["text"] for r in first]
    assert parse_choice("B", 4) == 1 and parse_choice("(c) số", 4) == 2 and parse_choice("xyz", 4) is None
    assert parse_choice("D", 2) is None


def test_variant_dataset():
    rows = [l.split("\t") for l in open(ROOT / "lexicon" / "teencode_variants.tsv", encoding="utf-8") if not l.startswith("#")]
    assert len(rows) >= 10000
    assert all(len(r) == 5 for r in rows)


def test_gioitre_dataset_and_alias():
    rows = [l.split("\t") for l in open(ROOT / "lexicon" / "teencode_gioitre_vungmien.tsv", encoding="utf-8") if not l.startswith("#")]
    assert len(rows) >= 10000
    import datamate_pipeline as dp

    class B:
        terms = ["Đậu phộng rang 200g", "Bia Tiger lon", "Thịt heo quay"]
    h = dp.alias_hints("bán được bao nhiêu gói lạc rang và thịt lợn quay", B())
    assert any("đậu phộng" in x for x in h) and any("'heo'" in x for x in h)
    assert dp.alias_hints("bia tiger lon bán bao nhiêu", B()) == []


def test_scenarios_and_compliance():
    rows = [l for l in open(ROOT / "lexicon" / "error_scenarios.tsv", encoding="utf-8") if not l.startswith("#")]
    assert len(rows) - 1 >= 20000
    import yaml
    cat = yaml.safe_load(open(ROOT / "lexicon" / "error_catalog.yaml", encoding="utf-8"))
    assert sum(len(v) for v in cat.values()) >= 170
    assert all(e.get("nguon") in ("rulebook", "train", "chung", "quy_dinh") for g in cat.values() for e in g)
    gpl = ROOT / "lexicon" / "gpl"
    if gpl.exists():
        for f in ("teencode_variants_tudien.tsv", "teencode_gioitre_vungmien_tudien.tsv"):
            n = sum(1 for l in open(gpl / f, encoding="utf-8") if not l.startswith("#"))
            assert n >= 20000, f
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_compliance.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_rulebook_reading():
    import datamate_pipeline as dp
    rb = ("# R\n## 1. Đơn\n| Luật | Mặc định |\n|---|---|\n| Tỉ lệ đơn huỷ | phần trăm đơn huỷ trong mọi đơn |\n"
          "## 2. Thời gian\n| Tuần | bắt đầu **Chủ nhật** |\n| Tết năm Y | từ **20 tháng Chạp** đến hết **mùng 7 tháng Giêng** |\n"
          "### Cà phê\n| Size | tên kết thúc (M)/(L) |\n- Bằng nhau xếp theo mã\n")
    ri = dp.RuleIndex(rb)
    assert len(ri.rules) == 5 and ri.rules[0]["id"] == "§1.1"
    assert ri.select("tỉ lệ huỷ tháng này", "")[0]["id"] == "§1.1"
    assert ri.select("size L bán bao nhiêu", "Quán cà phê")[0]["tieu_de"] == "Cà phê"
    p, w = dp.extract_rule_params(rb)
    assert p == {"tet_start": (12, 20), "tet_end": (1, 7), "week_start": 6} and not w


def test_status_groups():
    import datamate_pipeline as dp
    want = {"Đang vận chuyển": "dang_giao", "Không giao được": "huy", "Hoàn thành": "hoan_thanh", "Hoàn trả": "tra_hang",
            "hủy đơn": "huy", "đã lấy": "hoan_thanh", "Đã hoàn hàng": "tra_hang", "cancelled": "huy", "shipping": "dang_giao",
            "Đổi trả": "tra_hang", "Đang giao dịch": "dang_giao", "abc xyz": "chua_ro"}
    for v, g in want.items():
        assert dp.status_group(v) == g, (v, dp.status_group(v))


def test_readme_conversion_multilevel():
    import datamate_pipeline as d
    rd = "- Sữa TH 180ml: 1 thùng = 4 lốc\n- Sữa TH 180ml: 1 lốc = 4 hộp\n- Cát xây tô: 1 xe 4m3 = 4 m3\n"
    rc = d.readme_conversions(rd)
    tab = {(p, u): f for p, u, f, _ in rc}
    base = {(p, u): s for p, u, _, s in rc}
    final, b, multi = d._chain(tab, base)
    assert final[(d.key("Sữa TH 180ml"), "thung")] == 16 and multi == 1
    assert final[(d.key("Cát xây tô"), "xe 4m3")] == 4


def test_readme_cols_per_file_and_colnorm():
    import datamate_pipeline as d
    g, f = d.parse_readme_cols("## Cột trong `a.csv`\n- **Tiền**: số tiền\n## Cột trong `b.csv`\n- **Tiền**: tổng tiền đơn\n")
    assert f["a.csv"]["Tiền"] == "so tien" and f["b.csv"]["Tiền"] == "tong tien don"
    assert d._colnorm("SLNhapQuyDoi") == "sl nhap quy doi" and d._colnorm("IDCuaHang") == "id cua hang"
    assert d.clean_person_name("Từ Hải Thanh 0359470831") == "Từ Hải Thanh"


def test_total_row_not_person_name_and_sid():
    import numpy as np
    import datamate_pipeline as d
    for t in ["Tổng", "Tổng cộng", "TỔNG:", "Cộng ngày 20/05/2025", "Tổng ngày 01/03", "Total"]:
        assert d.TOTAL_ROW.match(t), t
    for t in ["Tong Huong Chau", "Tống Văn A", "Cong ty ABC"]:
        assert not d.TOTAL_ROW.match(t), t
    assert d.sid(1.0) == "1" and d.sid(" A1 ") == "A1" and d.sid(np.nan) != d.sid(np.nan)


def test_merge_same_structure_column_order_and_case():
    import collections
    import pandas as pd
    import datamate_pipeline as d
    T = {"a_1.csv": pd.DataFrame({"Mã đơn": ["1"], "DonVi": ["hộp"], "Tiền": [5.0]}),
         "a_2.csv": pd.DataFrame({"TIỀN": [7.0], "ma_don": ["2"], "DONVI": ["lon"]})}
    ty = {k: {c: "x" for c in v.columns} for k, v in T.items()}
    T2, _ = d.merge_same_structure(T, ty, collections.defaultdict(list))
    assert len(T2) == 1 and list(T2.values())[0]["Tiền"].tolist() == [5.0, 7.0]


def test_compiler_helpers_and_decide():
    import types
    import pandas as pd
    import compiler as C
    import datamate_pipeline as d
    assert C._months("doanh thu cac thang 11/2025, 12/2025 va 1/2026") == [(2025, 11), (2025, 12), (2026, 1)]
    assert C._months("tu thang 12/2025 toi thang 2/2026 thang nao") == [(2025, 12), (2026, 1), (2026, 2)]
    assert C.same_answer({"type": "string", "answer": "Cà phê sữa (M)"}, {"type": "string", "answer": "ca phe sua (m)"})
    assert not C.same_answer({"type": "number", "answer": 5}, {"type": "number", "answer": 6})
    assert C.same_answer({"type": "table", "answer": {"columns": ["a", "b"], "rows": [["x", 1], ["y", 2]]}},
                         {"type": "table", "answer": {"columns": ["a", "b"], "rows": [["y", 2], ["x", 1]]}})
    biz = types.SimpleNamespace(V={"hang": pd.DataFrame({"ma_hang": ["A1"], "ten_hang": ["Bia Tiger lon"]})}, readme="", data_end=None, T={})
    sp = C.parse_llm_spec('xx {"intent": "ton", "X": "2026-05-01", "mat_hang": "bia tiger lon", "doc_khac": []} yy', biz, "2026-05-01")
    assert sp["intent"] == "ton" and sp["mat_hang"] == "Bia Tiger lon" and sp["X"] == pd.Timestamp("2026-05-01")
    assert C.parse_llm_spec('{"intent": "bia"}', biz, "2026-05-01") is None
    item = {"q": "bao nhieu don", "q_norm": "bao nhiêu đơn", "expect_type": "number", "as_of": "2026-05-01"}
    assert d.sanity_issue({"type": "number", "answer": -3}, item, biz)
    assert d.sanity_issue({"type": "number", "answer": 2.5}, item, biz)
    assert d.sanity_issue({"type": "string", "answer": "x"}, item, biz)
    assert d.sanity_issue({"type": "number", "answer": 7}, item, biz) is None
    # quyết định: máy tính tin cậy cao thắng khi code lệch; tin cậy thấp + lệch + model giả -> theo đường code cũ
    args = types.SimpleNamespace(policy="strict", on_single="answer", llm="dummy", ask_choice="on")
    it = {"C": {"type": "number", "answer": 10}, "spec": {"intent": "so_don", "_nguon": "luat", "tin_cay": "cao"}, "_biz": biz,
          "q": "", "q_norm": "", "as_of": "2026-05-01"}
    res = {"A": {"line": {"type": "number", "answer": 11}}, "B": {"line": {"type": "number", "answer": 11}}}
    dec, fin, _ = d.decide_all(None, it, res, 2, args)
    assert dec == "compiler_wins" and fin["answer"] == 10
    it["spec"]["tin_cay"] = "thap"
    dec, fin, _ = d.decide_all(None, it, res, 2, args)
    assert dec == "agree" and fin["answer"] == 11


def test_phogpt_scorer_logic_with_fake_lm():
    """resolve_unsure_lm chỉ CHỌN trong phương án có sẵn; chênh điểm nhỏ thì giữ lựa chọn cũ; combine cần hai bên đồng ý."""
    import vn_lm_scorer as VS

    class FakeCN:
        def normalize(self, raw, forced=None):
            forced = forced or {}
            w = forced.get(0, "a")
            return {"text": f"{w} x", "unsure": [(0, "?", ["a", "b"])]}

    class FakeLM:
        def __init__(self, pref):
            self.pref = pref

        def logprob(self, ss):
            return [self.pref.get(s, 0.0) for s in ss]
    out, log = VS.resolve_unsure_lm(FakeCN(), ["q"], FakeLM({"b x": 5.0}), margin=1.0)
    assert out[0]["text"] == "b x" and log[0]["chon"] == "b" and log[0]["slot"] == 0
    out, log = VS.resolve_unsure_lm(FakeCN(), ["q"], FakeLM({"b x": 0.5}), margin=1.0)
    assert out[0]["text"] == "a x"
    assert VS.combine([{"q": 0, "slot": 0, "chon": "b"}], [{"q": 0, "slot": 0, "chon": "b"}]) == {0: {0: "b"}}
    assert VS.combine([{"q": 0, "slot": 0, "chon": "b"}], [{"q": 0, "slot": 0, "chon": "a"}]) == {}


def test_prob_abstain_and_repair_hint():
    import datamate_pipeline as d
    log = {"decision": "agree", "paths": {"A": {"line": {"type": "number", "answer": 1}, "attempts": 1},
                                          "B": {"line": {"type": "number", "answer": 1}, "attempts": 2}},
           "spec": {"_nguon": "llm"}, "final": {"type": "number", "answer": 1}, "normalize_changes": [1, 2]}
    f = d.answer_features(log)
    assert f["dec_agree"] == 1 and f["paths_same"] == 1 and f["repaired"] == 1 and f["norm_changes"] == 2
    m = {"w": {"bias": -2.0, "paths_same": 1.0}, "nguong": 0.5}
    assert d.prob_correct(m, log) < 0.5
    m["w"]["dec_agree"] = 3.0
    assert d.prob_correct(m, log) > 0.5
    import types
    B = types.SimpleNamespace(T={"x.csv": __import__("pandas").DataFrame({"Ngày bán": [1]})}, V={})
    h = d.repair_hint("KeyError: 'Ngay ban'", "T['x.csv']['Ngay ban']", B, {})
    assert "Ngày bán" in h
    it = {"q": "doanh thu tháng 3/2025", "q_norm": "doanh thu tháng 3/2025", "as_of": "2025-06-01",
          "P": d.resolve_periods("doanh thu tháng 3/2025", "2025-06-01")}
    assert d.rule_tests_for(it, types.SimpleNamespace(rule_tests="on")) == ["tuong_lai", "huy"]
    assert str(it["_seed_date"].date()) == "2025-03-16"
    it2 = dict(it, q="tỉ lệ huỷ tháng 3/2025", q_norm="tỉ lệ huỷ tháng 3/2025")
    assert d.rule_tests_for(it2, types.SimpleNamespace(rule_tests="on")) == ["tuong_lai"]


# ---------------------------------------------------------------- bổ sung (đợt rà rủi ro 09/11)
def test_canon_period_text_shorthand():
    import compiler as C
    A = "2026-07-08"
    cases = {
        "dthu t5/2026": "thang 5/2026", "th6/26 so don": "thang 6/2026", "5-2026 khach": "thang 5/2026",
        "01/2026 tat ca": "thang 1/2026", "tu 12/25 den 2/26": "tu thang 12/2025 den thang 2/2026",
        "th 11/25 dthu": "thang 11/2025", "thang 5 nam ngoai": "thang 5/2025", "q2/25 dthu": "quy 2/2025",
        "thang 8 dthu": "thang 8/2025",          # tháng 8 chưa tới (hôm nay 08/07/2026) -> năm trước (rulebook §2)
        "thang 7 dthu": "thang 7/2026",
    }
    for t, want in cases.items():
        assert want in C.canon_period_text(t, A), (t, C.canon_period_text(t, A))
    # KHÔNG được đổi: ngày đủ, giờ, 'T5' trơn (thứ Năm?), 'tháng 2 lần', 'ngày 5 tháng 6', số đếm
    for t in ["ngay 14/06/2026 gia", "ca sang 06:00-13:59", "t5 doanh thu", "thang 2 lan", "ngay 5 thang 6 dthu", "hop 10-12 cai"]:
        assert C.canon_period_text(t, A) == t, (t, C.canon_period_text(t, A))
    P = C._period(C.canon_period_text("quy 3 doanh thu", A), A)
    assert str(P[0].date()) == "2026-07-01" and str(P[1].date()) == "2026-07-08"
    P = C._period("tu dau nam toi nay", A)
    assert str(P[0].date()) == "2026-01-01" and str(P[1].date()) == "2026-07-08"


def test_week_start_param_used_by_compiler():
    import compiler as C
    old = C.WEEK_START
    try:
        C.WEEK_START = 0
        a, b = C._period("tuan truoc", "2025-06-25")        # thứ Tư
        assert (str(a.date()), str(b.date())) == ("2025-06-16", "2025-06-22")
        C.WEEK_START = 6                                      # rulebook đổi: tuần bắt đầu Chủ nhật
        a, b = C._period("tuan truoc", "2025-06-25")
        assert (str(a.date()), str(b.date())) == ("2025-06-15", "2025-06-21")
    finally:
        C.WEEK_START = old


def test_tie_oracle():
    import compiler as C
    assert C._tie_at([5, 5, 3], 1) and not C._tie_at([6, 5, 5], 1) and C._tie_at([9, 7, 7, 1], 2) and not C._tie_at([3, 2], 2)


def test_rule_delta_guard():
    import rule_delta as R
    base = "## 1. Đơn\n| Luật | Mặc định |\n|---|---|\n| Số đơn | chỉ tính đơn không huỷ, không trả |\n| Đơn bị huỷ | đơn có trạng thái huỷ |\n"
    fp = R.build_fingerprint(base)
    import json, tempfile, os
    p = os.path.join(tempfile.mkdtemp(), "fp.json")
    json.dump(fp, open(p, "w"))
    same = R.RuleDelta(base, p)
    assert same.block_reason({"intent": "so_don"}, "so don thang 5") is None
    ext = R.RuleDelta(base.replace("không trả |", "không trả. Kể cả đơn đặt qua sàn |"), p)
    assert ext.status["so don"] == "extended" and ext.block_reason({"intent": "so_don"}, "so don") is None
    chg = R.RuleDelta(base.replace("không huỷ, không trả", "kể cả đơn trả"), p)
    assert chg.status["so don"] == "changed" and chg.block_reason({"intent": "so_don"}, "so don") is not None
    rem = R.RuleDelta(base.replace("| Số đơn | chỉ tính đơn không huỷ, không trả |\n", ""), p)
    assert "so don" in rem.removed and rem.block_reason({"intent": "so_don"}, "so don") is not None
    new = R.RuleDelta(base + "| Khách sộp nhất | khách có giá trị trung bình một đơn cao nhất |\n", p)
    assert new.block_reason({"intent": "khach_mua_nhieu_tien_nhat"}, "khach sop nhat thang 5 la ai") is not None


def test_official_string_norm_and_snap():
    import datamate_pipeline as d
    assert d._snorm("Coca-Cola  330ml!") == "coca cola 330ml"
    assert d._snorm("CocaCola") != d._snorm("Coca Cola")          # BTC không xoá khoảng trắng
    g = {"type": "string", "answer": "Sting dâu 330ml", "alias": ["SP0001"]}
    assert d.score_one({"type": "string", "answer": "sting dâu-330ml"}, g) == 1
    assert d.score_one({"type": "string", "answer": "Stingdâu330ml"}, g) == -1

    class B:  # shop giả tối thiểu
        T = {"x": __import__("pandas").DataFrame({"ten": ["Sting dâu 330ml", "Bia Tiger lon"]})}
        V = {}
    b = B()
    assert d.snap_name("Sting dau 330ml", b)[0] == "Sting dâu 330ml"   # bỏ dấu khớp DUY NHẤT -> lấy cách viết trong file
    assert d.snap_name("Sting dâu 330ml", b)[1] is None
    assert d.snap_name("Pepsi", b)[0] == "Pepsi"
    assert d.clean_final({"type": "number", "answer": 3, "_tie": True}) == {"type": "number", "answer": 3}


def test_quarter_slash_year_not_month():
    """Hồi quy: 'quý 2/2026' từng bị viết lại thành 'quy thang 2/2026' (tháng 2) -> sai số liệu cả quý."""
    import datetime as dt
    import compiler as C
    a = dt.date(2026, 7, 9)
    for q, want in [("Cho hỏi tổng doanh thu quý 2/2026 là bao nhiêu?", ("2026-04-01", "2026-06-30")),
                    ("số đơn quys 3/2025 là nhiêu", ("2025-07-01", "2025-09-30")),
                    ("e so don hang quy 4/2025 la bao nhiu vay", ("2025-10-01", "2025-12-31")),
                    ("số đơn q1/2026", ("2026-01-01", "2026-03-31")),
                    ("doanh thu 5/2026", ("2026-05-01", "2026-05-31"))]:
        P = C._period(C.canon_period_text(C.key(q), a), a)
        assert P is not None and (str(P[0].date()), str(P[1].date())) == want, (q, P)
    assert C._period(C.canon_period_text(C.key("quyết định 3 món"), a), a) is None
