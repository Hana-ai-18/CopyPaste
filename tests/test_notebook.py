"""The notebook must be valid, every code cell must compile, and the smoke path must really execute."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "datamate_kaggle.ipynb"


def test_notebook_is_up_to_date_and_valid():
    before = NB.read_text(encoding="utf-8") if NB.exists() else ""
    subprocess.run([sys.executable, str(ROOT / "notebooks" / "build_notebook.py")], check=True, cwd=ROOT, capture_output=True)
    nb = nbformat.read(NB, as_version=4)
    nbformat.validate(nb)
    assert before == "" or before == NB.read_text(encoding="utf-8"), "datamate_kaggle.ipynb was stale - rebuilt; commit the new file"
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            compile(c.source, f"cell{i}", "exec")                       # syntax check (the notebook uses no magics)
            assert not re.search(r"^\s*[!%]", c.source, re.M), "no ! or % magics (keeps cells testable)"


def test_notebook_smoke_executes_end_to_end(tmp_path):
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb = nbformat.read(NB, as_version=4)
    cfg = next(c for c in nb.cells if c.cell_type == "code" and "GITHUB_REPO" in c.source)
    src = cfg.source
    for name, val in {"REPO_DIR": str(ROOT), "WORK": str(tmp_path), "RUN_MODE": "smoke", "RUN_TESTS": "False"}.items():
        src = re.sub(rf"^{name}\s*=.*$", f"{name} = {val!r}" if isinstance(val, str) else f"{name} = {val}", src, flags=re.M)
    cfg.source = src
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    text = "\n".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert "SMOKE TEST OK" in text and "Penalized Accuracy" in text
    assert "RUN_MODE = smoke" in text


@pytest.mark.parametrize("mode", ["preflight", "train_eval", "build_exemplars", "public_test"])
def test_notebook_real_modes_with_mock_llm(mode, demo_split, tmp_path_factory):
    """Executes the non-smoke branches (data discovery, run/resume, evaluate, exemplars, validation) with a scripted LLM."""
    import json
    import shutil

    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    base = tmp_path_factory.mktemp("nbmodes")
    repo = base / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "models", ".git", "*.jsonl"))
    (repo / "exemplars").mkdir(exist_ok=True)
    root = base / "dataroot"
    shutil.copytree(demo_split, root / "train")
    shutil.copytree(demo_split, root / "public_test", ignore=shutil.ignore_patterns("answers.jsonl"))
    model = base / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    cfg_file = base / "mock.json"
    cfg_file.write_text(json.dumps({"backend": "mock", "mock_responses": str(demo_split / "mock_responses.json")}))

    nb = nbformat.read(NB, as_version=4)
    # no network / heavy installs in the test: skip the GPU requirements install only
    deps = next(c for c in nb.cells if c.cell_type == "code" and "requirements-gpu.txt" in c.source)
    deps.source = deps.source.replace('run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-gpu.txt"])', "pass")
    cfg = next(c for c in nb.cells if c.cell_type == "code" and "GITHUB_REPO" in c.source)
    vals = {"REPO_DIR": str(repo), "WORK": str(base / "work"), "RUN_MODE": mode, "RUN_TESTS": False, "MODEL_DIR": str(model),
            "CONFIG": str(cfg_file), "DATA_ROOT": str(root), "EXEMPLAR_N": 4, "EVAL_N": 9, "EXEMPLAR_SCOPE": "all", "HOLDOUT_FOLD": "0/2", "NUM_PROCS": 2, "STAGGER_S": 0}
    src = cfg.source
    for name, val in vals.items():
        src = re.sub(rf"^{name}\s*=.*$", f"{name} = {val!r}", src, flags=re.M)
    cfg.source = src
    nbclient.NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(repo)}}).execute()
    text = "\n".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert "SMOKE TEST OK" in text and "splits found" in text
    if mode == "preflight":
        assert "==== PREFLIGHT ====" in text and (base / "work" / "auto_tune.json").exists()
    elif mode == "train_eval":
        assert "Penalized Accuracy" in text and "threshold_answer=" in text
    elif mode == "build_exemplars":
        assert (repo / "exemplars" / "exemplars.jsonl").read_text(encoding="utf-8").strip()
    else:
        assert "OK - submission looks valid" in text and "[shard 0/2]" in text and "[shard 1/2]" in text
        lines = [json.loads(l) for l in (base / "work" / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 13 and all("type" in r for r in lines)
