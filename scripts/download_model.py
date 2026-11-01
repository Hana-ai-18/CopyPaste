"""Download the model declared in configs/model.json (weights are NOT stored in git: 100 MB/file limit).

    python scripts/download_model.py --out models/llm            # needs internet, once
    python scripts/download_model.py --out models/llm --pin      # also write the resolved commit hash into configs/model.json

* skips the download when <out>/config.json already exists (use --force to redo it)
* afterwards counts the parameters from the safetensors headers and checks the 15B limit
* optional HF token (private/gated repos only) from the env var HF_TOKEN
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datamate.model_check import count_params  # noqa: E402

PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "*.tiktoken", "*.jinja"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/llm")
    ap.add_argument("--model_id", default=None, help="default: configs/model.json -> model_id")
    ap.add_argument("--revision", default=None, help="commit hash / tag; default: configs/model.json -> revision (null = latest)")
    ap.add_argument("--meta", default=str(ROOT / "configs" / "model.json"))
    ap.add_argument("--pin", action="store_true", help="write the downloaded commit hash into the meta file (reproducible builds)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit_b", type=float, default=15.0)
    a = ap.parse_args(argv)

    meta_path = Path(a.meta)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    model_id = a.model_id or meta.get("model_id")
    # the pinned revision belongs to the model in the meta file: never apply it to a different --model_id
    revision = a.revision or (meta.get("revision") if model_id == meta.get("model_id") else None) or None
    if not model_id:
        print("no model id: pass --model_id or set model_id in configs/model.json")
        return 2
    out = Path(a.out)

    if (out / "config.json").exists() and not a.force:
        print(f"model already present in {out} (use --force to download again)")
    else:
        from huggingface_hub import snapshot_download

        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {model_id} (revision={revision or 'latest'}) -> {out}", flush=True)
        snapshot_download(repo_id=model_id, revision=revision, local_dir=str(out), token=os.environ.get("HF_TOKEN") or None, allow_patterns=PATTERNS)

    if a.pin:
        try:
            from huggingface_hub import HfApi

            sha = HfApi().model_info(model_id, revision=revision, token=os.environ.get("HF_TOKEN") or None).sha
            if model_id != meta.get("model_id"):
                print("not writing the pin: --model_id differs from configs/model.json")
                raise RuntimeError("pin skipped")
            meta["revision"] = sha
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"pinned revision {sha} in {meta_path} - commit this file")
        except Exception as e:  # noqa: BLE001
            print(f"could not resolve the commit hash ({type(e).__name__}: {e}); not pinned")

    files = [p for p in out.rglob("*") if p.is_file()]
    print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e9:.1f} GB in {out}")
    info = count_params(out)
    if info["n_files"] == 0:
        print("WARNING: no .safetensors found - cannot verify the parameter count")
        return 0
    ok = info["params"] / 1e9 <= a.limit_b
    print(f"parameters: {info['params_b']:.2f}B -> {'OK' if ok else 'TOO BIG'} (limit {a.limit_b:g}B)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
