import datetime as dt
from datamate.intent import audit_code, resolve_periods, rule_checklist


def test_periods_explicit_and_relative():
    assert "2025-10-01 .. 2025-12-31" in resolve_periods("Quý 4 năm 2025 bao nhiêu đơn", "2026-06-24")[0]
    assert len(resolve_periods("Quy 1/2026 tong cong bao nhieu don", "2026-06-24")) == 1       # no phantom 'tháng 1/2026'
    assert resolve_periods("Hôm qua bán bao nhiêu", "2025-03-01") == ["- hôm qua = 2025-02-28"]
    r = resolve_periods("so voi tuan truoc, tuan nay", dt.date(2026, 2, 11))
    assert any("2026-02-02 .. 2026-02-08" in x for x in r) and any("2026-02-09 .. 2026-02-11" in x for x in r)
    assert "2026-05-01 .. 2026-05-24" in resolve_periods("Tháng này bán mấy ly", "2026-05-24")[0]


def test_periods_no_year_rule():
    assert "2025-11-01 .. 2025-11-30" in resolve_periods("thang 11 doanh thu", "2026-02-10")[0]     # November is after Feb 2026 -> last year
    assert "2026-01-01 .. 2026-03-31" in resolve_periods("quy 1 doanh thu", "2026-02-10")[0]


def test_periods_comparison_baselines():
    r = resolve_periods("Doanh thu th8/2025 so voi thang lien truoc", "2026-05-24")
    assert any("2025-07-01 .. 2025-07-31" in x for x in r)
    r = resolve_periods("Thang 2/26 so voi cung thang nam truoc", "2026-05-24")
    assert any("2025-02-01 .. 2025-02-28" in x for x in r)


def test_data_range_flag():
    r = resolve_periods("Tháng 3/2024 doanh thu", "2026-05-24", span=("2025-01-01", "2026-05-24"))
    assert "ENTIRELY OUTSIDE" in r[0]


def test_checklist():
    assert any("CANCEL RATE" in c for c in rule_checklist("T2/2026 don huy chiem bao nhieu phan tram"))
    assert any("REVENUE" in c for c in rule_checklist("Doanh thu hôm qua"))
    assert rule_checklist("xin chào") == []


def test_audit():
    bad = "df = ban_hang\nprint(df['thanh_tien'].sum())"
    assert audit_code("Doanh thu hôm qua", bad, "number", 100)
    good = "s = norm_status(df['Trạng thái'])\nd = df[(s=='hoan_thanh') & (df.d == as_of)]"
    assert audit_code("Doanh thu hôm qua", good, "number", 100) == []
    assert audit_code("Tỉ lệ huỷ T2/2026", "x = 0.02 / as_of\n", "percent", 0.02)         # fraction instead of percent
    assert audit_code("Doanh thu tháng 3/2026", good, "number", 100.5)                      # money not whole


def test_audit_dedupe_then_sum_line_amount():
    from datamate.intent import line_columns_from_profile
    prof = "- MONEY per LINE ['Tổng tiền']: revenue = sum over ALL lines ..."
    cols = line_columns_from_profile(prof)
    assert cols == {"Tổng tiền"}
    bad = "t = df[ok]\nper = t.drop_duplicates('Mã đơn hàng')\nanswer = to_num(per['Tổng tiền']).sum()\nnorm_status as_of"
    assert audit_code("doanh thu hôm nay", bad, "number", 100, cols)
    good = "per = order_amounts(t, 'Mã đơn hàng', 'Tổng tiền')\nanswer = per['amount'].sum()\nnorm_status as_of"
    assert audit_code("doanh thu hôm nay", good, "number", 100, cols) == []


def test_crosscheck_disagreement_uses_engine():
    from datamate.agent import AgentResult, DataMateAgent

    class SB:
        def __init__(self, v):
            self.v = v

        def ref(self, q, a, timeout=60.0):
            return ("number", self.v) if self.v is not None else None

    ag = DataMateAgent.__new__(DataMateAgent)
    ag.fast_mode = False

    def mk():
        return AgentResult({"type": "number", "answer": 100}, "answer", 1.0, None, 3, {}, [], 0.0)

    ag.sb = SB(100)
    r = ag._crosscheck(mk(), "doanh thu hom nay", "2026-01-01")
    assert r.decision == "answer" and r.ref == {"answer": 100, "agree": True, "used": False}
    ag.sb = SB(250)
    r = ag._crosscheck(mk(), "doanh thu hom nay", "2026-01-01")
    assert r.decision == "answer" and r.record == {"type": "number", "answer": 250} and r.ref["agree"] is False
    ag.sb = SB(None)
    assert ag._crosscheck(mk(), "x", "2026-01-01").decision == "answer"


def test_crosscheck_fills_abstain_and_skips_unanswerable():
    from datamate.agent import AgentResult, DataMateAgent

    class SB:
        def ref(self, q, a, timeout=60.0):
            return ("number", 7)

    ag = DataMateAgent.__new__(DataMateAgent)
    ag.fast_mode, ag.sb = False, SB()
    r = ag._crosscheck(AgentResult({"type": "abstain"}, "abstain", 0.2, None, 3, {}, [], 0.0), "doanh thu hom nay", "2026-01-01")
    assert r.record == {"type": "number", "answer": 7} and r.decision == "answer"
    u = AgentResult({"type": "unanswerable", "reason": "x"}, "unanswerable", 1.0, None, 3, {}, [], 0.0)
    assert ag._crosscheck(u, "doanh thu hom nay", "2026-01-01").decision == "unanswerable"
