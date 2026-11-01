"""Configuration: defaults + JSON file + `--set a.b=value` overrides."""
from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULT_CONFIG: dict = {
    "seed": 0,
    "backend": "hf",                 # hf | vllm | mock
    "model_path": "models/llm",      # folder with the weights inside the Docker image
    "max_model_params_b": 15,        # compliance guard (billions of parameters, all models together)
    "time_budget_s": 18900,          # 5h15m: hard stop for the whole run (limit is 6h incl. model loading)
    "hf": {
        "load_in_4bit": True,
        "dtype": "float16",          # T4 has no bf16; use bfloat16 on RTX 5090 / A100 if you prefer
        "batch_size": 1,             # >1 can change numerics -> keep 1 for reproducibility
        "max_input_tokens": 12000,
        "trust_remote_code": False,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "vllm": {
        "dtype": "auto",
        "quantization": None,        # e.g. "awq" for AWQ checkpoints
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.88,
        "max_model_len": 16384,
        "enforce_eager": False,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "generation": {"max_new_tokens": 900},
    "agent": {
        "stages": [3, 5],            # run 3 variants; if they all agree stop, else run up to 5
        "max_repairs": 2,
        "exec_timeout_s": 60,
        "threshold_answer": 0.6,     # min vote share to answer (else abstain)  -> calibrate with evaluate.py
        "threshold_unanswerable": 0.8,
        "per_question_time_s": 150,
        "min_time_per_question_s": 20,   # below this average budget the agent switches to fast mode
        "n_exemplars": 3,
        "exemplars_path": "exemplars/exemplars.jsonl",
        "max_profile_chars": 7000,
        "max_rulebook_chars": 4500,
        "max_readme_chars": 2500,
    },
}


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            cfg = _merge(cfg, json.load(f))
    for item in overrides or []:
        key, _, raw = item.partition("=")
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg
