import json
import os

import pytest

from datamate.agent import DataMateAgent, BusinessContext, extract_code, to_submission
from datamate.config import load_config
from datamate.llm import MockBackend
from datamate.retrieval import ExemplarBank

AS_OF = "2026-03-04"


@pytest.fixture()
def loaded(sandbox, demo_split):  # function-scoped: every test starts from biz_demo1 (other tests load other businesses)
    info = sandbox.load(demo_split / "data" / "biz_demo1", "biz_demo1", demo_split / "rulebook.md", demo_split / "calendar.csv")
    return sandbox, info


# ----------------------------------------------------------------------------- sandbox
def test_load_info(loaded):
    _, info = loaded
    assert set(info.tables) == {"products", "customers", "orders", "order_items"} and info.lunar_ok
    assert "completed" in info.profile


def test_basic_answer_and_helpers(loaded, demo_split):
    import pandas as pd

    sb, _ = loaded
    r = sb.run("answer_type='number'\nanswer=round_vnd(len(orders))", AS_OF)
    assert r.ok and r.answer == len(pd.read_csv(demo_split / "data" / "biz_demo1" / "orders.csv"))
    r = sb.run("s,e = last_week(as_of)\nprint(s, e)\nanswer_type='date'\nanswer=str(s)", AS_OF)
    assert r.ok and r.answer == "2026-02-23" and "2026-03-01" in r.stdout
    r = sb.run("answer_type='date'\nanswer=tet_window(2026)[0]", AS_OF)
    assert r.ok and r.answer < "2026-02-17"
    r = sb.run("answer_type='number'\nanswer=sql('select count(*) as n from orders').n.iloc[0]", AS_OF)
    assert r.ok and r.answer > 100


def test_errors_are_reported_with_line_info(loaded):
    sb, _ = loaded
    r = sb.run("x = 1\nanswer_type='number'\nanswer = orders['nope'].sum()", AS_OF)
    assert not r.ok and "KeyError" in r.error and "line 3" in r.error
    r = sb.run("answer = 5", AS_OF)
    assert not r.ok and "answer_type" in r.error
    r = sb.run("answer_type='number'\nanswer = float('nan')", AS_OF)
    assert not r.ok and "NaN" in r.error
    r = sb.run("answer_type='number'\nanswer = 'abc'", AS_OF)
    assert not r.ok and "number" in r.error
    r = sb.run("def broken(:\n  pass", AS_OF)
    assert not r.ok and "SyntaxError" in r.error


def test_restrictions_and_timeout(loaded):
    sb, _ = loaded
    for code in ("import os", "import subprocess", "import socket", "__import__('sys')", "open('/etc/passwd')"):
        r = sb.run(code, AS_OF)
        assert not r.ok, code
    r = sb.run("while True: pass", AS_OF, timeout=2)
    assert not r.ok and "Timeout" in r.error and r.elapsed < 8
    # the worker is still healthy after a timeout
    assert sb.run("answer_type='number'\nanswer=1", AS_OF).ok


def test_runs_are_isolated(loaded):
    sb, _ = loaded
    n = sb.run("answer_type='number'\nanswer=len(orders)", AS_OF).answer
    sb.run("orders.drop(orders.index, inplace=True)\nglobal_leak = 1\nanswer_type='number'\nanswer=len(orders)", AS_OF)
    assert sb.run("answer_type='number'\nanswer=len(orders)", AS_OF).answer == n                      # data not corrupted
    assert not sb.run("answer_type='number'\nanswer=global_leak", AS_OF).ok                           # no state leaks between runs


def test_unanswerable_and_output_cap(loaded):
    sb, _ = loaded
    r = sb.run("print('x'*10000)\ndeclare_unanswerable('no cost column')", AS_OF)
    assert r.ok and r.answer_type == "unanswerable" and r.reason == "no cost column" and len(r.stdout) <= 3000


def test_other_business_replaces_data(sandbox, demo_split):
    sandbox.load(demo_split / "data" / "biz_demo3", "biz_demo3", None, None)
    r = sandbox.run("answer_type='list'\nanswer=sorted(dfs)", AS_OF)
    assert r.ok and r.answer == ["batches", "medicines"]
    assert not sandbox.run("answer_type='number'\nanswer=len(orders)", AS_OF).ok


# ----------------------------------------------------------------------------- agent logic
def test_extract_code():
    assert extract_code("plan\n```python\nanswer_type='number'\nanswer=1\n```\nbye") == "answer_type='number'\nanswer=1"
    two = "```python\nx=1\n```\nthen\n```python\nanswer_type='number'\nanswer=2\n```"
    assert extract_code(two).endswith("answer=2")
    assert extract_code("<think>```python\nanswer=0\n```</think>```python\nanswer_type='number'\nanswer=3\n```").endswith("answer=3")
    assert extract_code("```python\nanswer_type='number'\nanswer=4") == "answer_type='number'\nanswer=4"     # unterminated fence
    assert extract_code("I cannot do that") is None


def _agent(sb, responder, **agent_over):
    cfg = load_config(None, [f"agent.{k}={json.dumps(v)}" for k, v in agent_over.items()] + ['backend="mock"'])
    return DataMateAgent(MockBackend(responder), sb, cfg, ExemplarBank([]))


def _fence(code):
    return f"```python\n{code}\n```"


def test_unanimous_stops_after_first_stage(loaded):
    sb, info = loaded
    calls = {"n": 0}

    def resp(msgs):
        calls["n"] += 1
        return _fence("answer_type='number'\nanswer=42")

    ag = _agent(sb, resp)
    r = ag.answer(BusinessContext("b", info), "", "câu hỏi", AS_OF)
    assert r.decision == "answer" and r.confidence == 1.0 and r.n_run == 3 and calls["n"] == 3
    assert to_submission("q1", r.record) == {"id": "q1", "type": "number", "answer": 42}


def test_vote_majority_and_threshold(loaded):
    sb, info = loaded

    def resp(msgs):
        v = int(msgs[1]["content"].split("Strategy variant: ")[1].split()[0])
        return _fence(f"answer_type='number'\nanswer={1 if v in (0, 3, 4) else 2}")

    ag = _agent(sb, resp)                       # variants 0..4 -> 1,2,2?,... -> 3 votes for "1" out of 5
    r = ag.answer(BusinessContext("b", info), "", "q", AS_OF)
    assert r.n_run == 5 and r.record["answer"] == 1 and r.confidence == pytest.approx(0.6) and r.decision == "answer"
    strict = _agent(sb, resp, threshold_answer=0.8)
    r2 = strict.answer(BusinessContext("b", info), "", "q", AS_OF)
    assert r2.decision == "abstain" and r2.record == {"type": "abstain"} and r2.top_record["answer"] == 1   # top kept for calibration


def test_repair_loop_and_failures_reduce_confidence(loaded):
    sb, info = loaded

    def resp(msgs):
        if len(msgs) > 2:                                           # repair turn
            assert "KeyError" in msgs[-1]["content"]
            return _fence("answer_type='number'\nanswer=7")
        return _fence("answer_type='number'\nanswer=orders['zzz'].sum()")

    ag = _agent(sb, resp, max_repairs=1)
    r = ag.answer(BusinessContext("b", info), "", "q", AS_OF)
    assert r.decision == "answer" and r.record["answer"] == 7 and all(c.repairs == 1 for c in r.candidates)
    never = _agent(sb, lambda m: _fence("answer = orders['zzz']"), max_repairs=1)
    r = never.answer(BusinessContext("b", info), "", "q", AS_OF)
    assert r.decision == "abstain" and r.record["type"] == "abstain"


def test_unanswerable_needs_higher_threshold(loaded):
    sb, info = loaded
    ag = _agent(sb, lambda m: _fence("declare_unanswerable('thiếu giá vốn')"))
    r = ag.answer(BusinessContext("b", info), "", "q", AS_OF)
    assert r.decision == "unanswerable" and to_submission("q", r.record) == {"id": "q", "type": "unanswerable", "reason": "thiếu giá vốn"}


def test_fast_mode_runs_single_stage_without_repairs(loaded):
    sb, info = loaded
    n = {"c": 0}

    def resp(msgs):
        n["c"] += 1
        return _fence("answer = 1")                                 # always invalid

    ag = _agent(sb, resp)
    ag.fast_mode = True
    r = ag.answer(BusinessContext("b", info), "", "q", AS_OF)
    assert n["c"] == 3 and r.decision == "abstain"                  # 3 variants, no repair round, no 2nd stage


def test_prompt_contains_rulebook_profile_and_question(loaded):
    sb, info = loaded
    seen = {}

    def resp(msgs):
        seen["m"] = msgs
        return _fence("answer_type='number'\nanswer=1")

    ag = _agent(sb, resp)
    ag.answer(BusinessContext("b", info), "RULEBOOK-MARKER revenue = ...", "doanh thu hôm nay?", AS_OF)
    sys_p, user = seen["m"][0]["content"], seen["m"][1]["content"]
    assert "declare_unanswerable" in sys_p and "rank_top" in sys_p and "last_week" in sys_p
    assert "RULEBOOK-MARKER" in user and "order_items" in user and "Question: doanh thu hôm nay?" in user and AS_OF in user
