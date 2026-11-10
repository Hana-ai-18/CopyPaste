"""End-to-end: the real CLI on the generated demo split with the scripted mock LLM."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from datamate.scoring import penalized_accuracy

ROOT = Path(__file__).resolve().parents[1]


def _read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _run(demo, out, *extra):
    cmd = [sys.executable, "-m", "datamate.run", "--data_dir", str(demo), "--out", str(out / "submission.jsonl"), "--logs", str(out / "logs"),
           "--config", str(ROOT / "configs" / "default.json"), "--set", 'backend="mock"', "--set", f'mock_responses="{demo / "mock_responses.json"}"', *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300, env={**__import__("os").environ, "PYTHONHASHSEED": "0"})


@pytest.fixture(scope="module")
def run1(demo_split, tmp_path_factory):
    out = tmp_path_factory.mktemp("run1")
    p = _run(demo_split, out, "--eval")
    assert p.returncode == 0, p.stderr[-2000:] + p.stdout[-1000:]
    return out, p


def test_every_question_has_a_valid_line(run1, demo_split):
    out, _ = run1
    qs = [q["id"] for q in _read(demo_split / "questions.jsonl")]
    sub = _read(out / "submission.jsonl")
    assert [r["id"] for r in sub] == qs                                  # same ids, same order
    v = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_submission.py"), "--sub", str(out / "submission.jsonl"),
                        "--questions", str(demo_split / "questions.jsonl")], capture_output=True, text=True)
    assert v.returncode == 0, v.stdout


def test_scores_match_design(run1, demo_split):
    out, p = run1
    sub = {r["id"]: r for r in _read(out / "submission.jsonl")}
    m = penalized_accuracy(sub, _read(demo_split / "answers.jsonl"))
    # 12 correct (incl. repaired code, minority-vote fix and the unanswerable one) + 1 deliberate abstain
    assert (m["correct"], m["wrong"], m["abstain"]) == (12, 0, 1) and m["PA"] == pytest.approx(12 / 13)
    assert "Penalized Accuracy" in p.stdout


def test_logs_have_what_the_organisers_require(run1):
    out, _ = run1
    logs = _read(out / "logs" / "log.jsonl")
    assert len(logs) == 13
    for l in logs:
        assert {"id", "submitted", "decision", "elapsed_s"} <= set(l)
    ok = [l for l in logs if l.get("candidates")]
    assert ok and all("code" in c and "stdout" in c for c in ok[0]["candidates"])         # generated code + execution output
    info = json.load(open(out / "logs" / "run_info.json", encoding="utf-8"))
    assert info["n_written"] == 13 and "total_seconds" in info and info["config"]["seed"] == 0


def test_mechanisms_visible_in_logs(run1):
    out, _ = run1
    by = {l["id"]: l for l in _read(out / "logs" / "log.jsonl")}
    assert any(c["repairs"] == 1 for c in by["q_demo_0001"]["candidates"])                   # broken code was repaired
    assert by["q_demo_0003"]["n_run"] == 5 and by["q_demo_0003"]["confidence"] == pytest.approx(0.8)  # stage 2 + vote
    assert by["q_demo_0009"]["decision"] == "abstain" and by["q_demo_0009"]["top_candidate"]["type"] == "number"
    assert by["q_demo_0008"]["decision"] == "unanswerable"


def test_deterministic_rerun(run1, demo_split, tmp_path):
    out1, _ = run1
    assert _run(demo_split, tmp_path).returncode == 0
    assert _read(out1 / "submission.jsonl") == _read(tmp_path / "submission.jsonl")


def test_resume_does_not_redo_work(run1, demo_split, tmp_path):
    out1, _ = run1
    (tmp_path / "logs").mkdir()
    sub = _read(out1 / "submission.jsonl")
    with open(tmp_path / "submission.jsonl", "w", encoding="utf-8") as f:         # pretend only the first 5 were done
        for r in sub[:5]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    assert _run(demo_split, tmp_path, "--resume").returncode == 0
    assert _read(tmp_path / "submission.jsonl") == sub
    assert len(_read(tmp_path / "logs" / "log.jsonl")) == 8                       # only the 8 missing ones were answered


def test_time_budget_exhausted_gives_abstain_for_everything(demo_split, tmp_path):
    p = _run(demo_split, tmp_path, "--set", "time_budget_s=0")
    assert p.returncode == 0, p.stderr[-1500:]
    sub = _read(tmp_path / "submission.jsonl")
    assert len(sub) == 13 and all(r == {"id": r["id"], "type": "abstain"} for r in sub)


def test_unknown_business_and_bad_question_do_not_crash(demo_split, tmp_path):
    qs = _read(demo_split / "questions.jsonl")[:2]
    qs.append({"id": "q_x", "business_id": "biz_missing", "question": "?", "as_of_date": "2026-03-04"})
    qfile = tmp_path / "q.jsonl"
    qfile.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in qs), encoding="utf-8")
    p = _run(demo_split, tmp_path, "--questions", str(qfile))
    assert p.returncode == 0, p.stderr[-1500:]
    sub = {r["id"]: r for r in _read(tmp_path / "submission.jsonl")}
    assert sub["q_x"] == {"id": "q_x", "type": "abstain"} and sub["q_demo_0001"]["type"] == "string"


def test_calibrate_cli(run1, demo_split):
    out, _ = run1
    p = subprocess.run([sys.executable, "-m", "datamate.evaluate", "--pred", str(out / "submission.jsonl"), "--gold", str(demo_split / "answers.jsonl"),
                        "--calibrate", str(out / "logs" / "log.jsonl")], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0 and "threshold_answer=" in p.stdout
