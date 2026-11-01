"""Count the parameters of a local model folder (reads only the safetensors headers - no torch needed).

    python -m datamate.model_check models/llm --limit 15

Used as a compliance guard: DataMate allows at most 15B parameters IN TOTAL across all models used at inference
(MoE models count ALL parameters). Caveats: tied weights stored once are counted once; int4/int8-packed checkpoints
under-count - check the bf16 original for those.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path


def count_params(model_dir: str | Path) -> dict:
    total, files = 0, []
    for f in sorted(Path(model_dir).rglob("*.safetensors")):
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name != "__metadata__":
                total += math.prod(meta["shape"])
        files.append(f.name)
    return {"params": total, "params_b": round(total / 1e9, 3), "n_files": len(files)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--limit", type=float, default=15.0, help="max total parameters in billions (DataMate: 15)")
    a = ap.parse_args(argv)
    info = count_params(a.model_dir)
    if info["n_files"] == 0:
        print(f"no .safetensors files in {a.model_dir} - cannot count parameters")
        return 2
    ok = info["params"] / 1e9 <= a.limit
    print(f"{a.model_dir}: {info['params_b']:.2f}B parameters in {info['n_files']} files -> {'OK' if ok else 'TOO BIG'} (limit {a.limit:g}B)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
