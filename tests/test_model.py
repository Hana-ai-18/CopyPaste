"""model_check (safetensors header counting) and scripts/download_model.py (huggingface_hub stubbed)."""
import importlib.util
import json
import struct
import sys
import types
from pathlib import Path

from datamate.model_check import count_params, main as check_main

ROOT = Path(__file__).resolve().parents[1]


def write_st(path: Path, shapes: dict):
    header = {k: {"dtype": "BF16", "shape": v, "data_offsets": [0, 0]} for k, v in shapes.items()}
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def test_count_params_and_limit(tmp_path):
    write_st(tmp_path / "a.safetensors", {"w": [1000, 1000], "b": [1000]})
    write_st(tmp_path / "b.safetensors", {"x": [10, 10]})
    info = count_params(tmp_path)
    assert info["params"] == 1_000_000 + 1000 + 100 and info["n_files"] == 2
    assert check_main([str(tmp_path)]) == 0
    assert check_main([str(tmp_path), "--limit", "0.0001"]) == 1


def test_count_params_no_files(tmp_path):
    assert check_main([str(tmp_path)]) == 2


def _load_script():
    spec = importlib.util.spec_from_file_location("download_model", ROOT / "scripts" / "download_model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_download_script(tmp_path, monkeypatch):
    calls = {}

    def snapshot_download(repo_id, revision, local_dir, token, allow_patterns):
        calls.update(repo_id=repo_id, revision=revision)
        Path(local_dir).mkdir(parents=True)
        (Path(local_dir) / "config.json").write_text("{}")
        write_st(Path(local_dir) / "model.safetensors", {"w": [100, 100]})

    class HfApi:
        def model_info(self, model_id, revision=None, token=None):
            return types.SimpleNamespace(sha="abc123")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download, HfApi=HfApi))
    meta = tmp_path / "model.json"
    meta.write_text(json.dumps({"model_id": "x/y", "revision": None}))
    dm = _load_script()
    out = tmp_path / "llm"
    assert dm.main(["--out", str(out), "--meta", str(meta), "--pin"]) == 0
    assert calls == {"repo_id": "x/y", "revision": None}
    assert json.loads(meta.read_text())["revision"] == "abc123"
    calls.clear()
    assert dm.main(["--out", str(out), "--meta", str(meta)]) == 0 and not calls     # already present -> no download
    assert dm.main(["--out", str(out), "--meta", str(meta), "--limit_b", "0.000001"]) == 1   # too big -> fails


def test_pinned_revision_not_applied_to_other_model(tmp_path, monkeypatch):
    seen = {}

    def snapshot_download(repo_id, revision, local_dir, token, allow_patterns):
        seen.update(repo_id=repo_id, revision=revision)
        Path(local_dir).mkdir(parents=True)
        (Path(local_dir) / "config.json").write_text("{}")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download, HfApi=None))
    meta = tmp_path / "m.json"
    meta.write_text(json.dumps({"model_id": "x/big", "revision": "deadbeef"}))
    dm = _load_script()
    dm.main(["--out", str(tmp_path / "o1"), "--meta", str(meta), "--model_id", "x/small"])
    assert seen == {"repo_id": "x/small", "revision": None}
    dm.main(["--out", str(tmp_path / "o2"), "--meta", str(meta)])
    assert seen == {"repo_id": "x/big", "revision": "deadbeef"}
