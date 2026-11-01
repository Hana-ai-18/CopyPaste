import json
import sys
import types

import numpy as np
import pytest

from datamate.config import DEFAULT_CONFIG, load_config
from datamate.retrieval import ExemplarBank
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ----------------------------------------------------------------------------- config
def test_config_overrides(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"agent": {"stages": [2]}, "hf": {"load_in_4bit": False}}))
    c = load_config(p, ["agent.threshold_answer=0.7", 'model_path="m/x"', "agent.stages=[1,2]", "backend=vllm"])
    assert c["agent"]["threshold_answer"] == 0.7 and c["agent"]["stages"] == [1, 2] and c["model_path"] == "m/x" and c["backend"] == "vllm"
    assert c["hf"]["load_in_4bit"] is False and c["hf"]["dtype"] == "float16"        # deep-merged with defaults
    assert DEFAULT_CONFIG["agent"]["stages"] == [3, 5]                                # defaults untouched


@pytest.mark.parametrize("name", ["default", "t4_kaggle", "rtx5090_vllm"])
def test_shipped_configs_load(name):
    c = load_config(ROOT / "configs" / f"{name}.json")
    assert c["agent"]["stages"][0] >= 1 and c["backend"] in ("hf", "vllm")


# ----------------------------------------------------------------------------- retrieval
def test_exemplar_bank_search():
    bank = ExemplarBank([{"question": "sp bán chạy nhất tuần rồi", "code": "a"}, {"question": "doanh thu tháng này", "code": "b"},
                         {"question": "ai còn nợ tiền", "code": "c"}, {"question": "sp bán chạy nhất tuần rồi", "code": "dup"}])
    r = bank.search("Sản phẩm bán chạy nhất tuần trước?", 2)
    assert r[0]["code"] == "a" and len({x["question"] for x in r}) == len(r)
    assert bank.search("zzz qqq", 3) == []
    assert ExemplarBank.load("/does/not/exist").search("x") == []


# ----------------------------------------------------------------------------- validate_submission
def test_validate_records():
    v = _load_script("validate_submission")
    assert v.check_record({"id": "1", "type": "number", "answer": 5}) == []
    assert v.check_record({"id": "1", "type": "abstain"}) == []
    assert v.check_record({"id": "1", "type": "unanswerable", "reason": "x"}) == []
    assert v.check_record({"id": "1", "type": "table", "answer": {"columns": ["a", "b"], "rows": [["x", 1]]}}) == []
    bad = [{"type": "number", "answer": "5"}, {"type": "number", "answer": True}, {"type": "date", "answer": "04/03/2026"},
           {"type": "boolean", "answer": "true"}, {"type": "list", "answer": "a"}, {"type": "string", "answer": ""},
           {"type": "table", "answer": {"columns": ["a"], "rows": [[1, 2]]}}, {"type": "mystery", "answer": 1}, {"type": "number"}]
    for r in bad:
        assert v.check_record(r), r


# ----------------------------------------------------------------------------- HF backend (stubbed torch/transformers)
def test_hf_backend_flow_with_stubs(monkeypatch):
    class NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    torch = types.ModuleType("torch")
    torch.float16 = "f16"
    torch.manual_seed = lambda s: None
    torch.no_grad = lambda: NoGrad()
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda s: None, empty_cache=lambda: None)

    class Enc(dict):
        def to(self, device): return self

    captured = {}

    class Tok:
        pad_token, eos_token, pad_token_id, padding_side = None, "</s>", 0, "right"

        @classmethod
        def from_pretrained(cls, path, **kw):
            captured["tok_kw"] = kw
            return cls()

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
            captured["template_kw"] = kw
            return "P:" + messages[-1]["content"]

        def __call__(self, prompts, **kw):
            if isinstance(prompts, str):
                return {"input_ids": list(range(len(prompts)))}
            width = max(len(p) for p in prompts)
            return Enc(input_ids=np.ones((len(prompts), width), dtype=int), attention_mask=np.ones((len(prompts), width), dtype=int))

        def decode(self, row, skip_special_tokens=True):
            return "".join(chr(int(c)) for c in row)

    class Model:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, path, **kw):
            captured["model_kw"] = kw
            return cls()

        def eval(self): return self

        def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, **kw):
            captured["gen"] = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
            add = np.array([[72, 105]] * input_ids.shape[0])                    # "Hi"
            return np.concatenate([input_ids, add], axis=1)

    tf = types.ModuleType("transformers")
    tf.__version__ = "4.57.0"
    tf.AutoTokenizer, tf.AutoModelForCausalLM, tf.BitsAndBytesConfig = Tok, Model, lambda **k: k
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", tf)

    from datamate.llm import HFBackend

    b = HFBackend("some/path", {"load_in_4bit": True, "dtype": "float16", "batch_size": 2, "chat_template_kwargs": {"enable_thinking": False}},
                  {"max_new_tokens": 33}, seed=1)
    assert b.generate([{"role": "user", "content": "xx"}]) == "Hi"
    assert b.generate_batch([[{"role": "user", "content": "a"}], [{"role": "user", "content": "bbb"}], [{"role": "user", "content": "c"}]]) == ["Hi"] * 3
    assert captured["gen"] == {"max_new_tokens": 33, "do_sample": False}           # greedy, deterministic
    assert captured["model_kw"]["dtype"] == "f16" and "quantization_config" not in captured["model_kw"]   # no CUDA in the stub
    assert captured["template_kw"] == {"enable_thinking": False}
    assert b.count_tokens([{"role": "user", "content": "abcd"}]) == len("P:abcd")
    assert b.tok.pad_token == "</s>" and b.tok.padding_side == "left"


def test_hf_backend_retries_after_cuda_oom(monkeypatch):
    import numpy as np

    class NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    freed = []
    torch = types.ModuleType("torch")
    torch.float16 = "f16"
    torch.manual_seed = lambda s: None
    torch.no_grad = lambda: NoGrad()
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, manual_seed_all=lambda s: None, empty_cache=lambda: freed.append(1))

    class Enc(dict):
        def to(self, device): return self

    seen = []

    class Tok:
        pad_token, eos_token, pad_token_id, padding_side = None, "</s>", 0, "right"

        @classmethod
        def from_pretrained(cls, path, **kw): return cls()
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw): return messages[-1]["content"]
        def __call__(self, prompts, **kw):
            seen.append((len(prompts), kw.get("max_length")))
            w = max(len(p) for p in prompts)
            return Enc(input_ids=np.ones((len(prompts), w), dtype=int), attention_mask=np.ones((len(prompts), w), dtype=int))
        def decode(self, row, skip_special_tokens=True): return "".join(chr(int(c)) for c in row)

    class Model:
        device, calls, kw = "cpu", 0, None

        @classmethod
        def from_pretrained(cls, path, **kw):
            cls.kw = kw
            return cls()
        def eval(self): return self
        def generate(self, input_ids, attention_mask, max_new_tokens, **kw):
            Model.calls += 1
            if input_ids.shape[0] > 1 or Model.calls == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return np.concatenate([input_ids, np.array([[72, 105]])], axis=1)

    tf = types.ModuleType("transformers")
    tf.__version__ = "4.57.0"
    tf.AutoTokenizer, tf.AutoModelForCausalLM, tf.BitsAndBytesConfig = Tok, Model, lambda **k: k
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", tf)
    from datamate.llm import HFBackend

    b = HFBackend("p", {"load_in_4bit": True, "dtype": "float16", "batch_size": 2, "max_input_tokens": 1000}, {"max_new_tokens": 5})
    assert Model.kw["device_map"] == {"": 0} and Model.kw["low_cpu_mem_usage"] is True      # whole model on the one visible GPU, streamed load
    assert b.generate_batch([[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]) == ["Hi", "Hi"]
    assert freed and (1, 750) in seen                                                       # cache freed, retried one by one with a 25 % shorter context
