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

    b = HFBackend("some/path", {"load_in_4bit": True, "dtype": "float16", "batch_size": 2, "prefill_chunk": 0, "stop_at_code_end": False, "chat_template_kwargs": {"enable_thinking": False}},
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

    b = HFBackend("p", {"load_in_4bit": True, "dtype": "float16", "batch_size": 2, "prefill_chunk": 0, "stop_at_code_end": False, "max_input_tokens": 1000}, {"max_new_tokens": 5})
    assert Model.kw["device_map"] == {"": 0} and Model.kw["low_cpu_mem_usage"] is True      # whole model on the one visible GPU, streamed load
    assert b.generate_batch([[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]) == ["Hi", "Hi"]
    assert freed and (1, 750) in seen                                                       # cache freed, retried one by one with a 25 % shorter context


def test_chunked_prefill_matches_full_prefill():
    """Chunked prefill (what keeps a 13k-token prompt from OOM-ing a T4) must give exactly the same greedy tokens as one big pass."""
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    import random
    from datamate.llm import HFBackend

    torch.manual_seed(0)
    cfg = tf.Qwen3Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
    model = tf.Qwen3ForCausalLM(cfg).eval()

    class BE(dict):
        def to(self, dev): return self

    class Tok:
        pad_token_id, eos_token_id = 0, 1

        def __call__(self, p, return_tensors=None, add_special_tokens=False, truncation=False, max_length=None, padding=False):
            ps = [p] if isinstance(p, str) else p
            t = torch.tensor([[2 + (ord(c) % 250) for c in x][-max_length:] for x in ps])
            return BE(input_ids=t, attention_mask=torch.ones_like(t))

        def decode(self, row, skip_special_tokens=True): return ",".join(map(str, row.tolist()))

    b = HFBackend.__new__(HFBackend)
    b.torch, b.model, b.tok, b.max_in = torch, model, Tok(), 4000
    b.prefix_cache, b._pc, b._pc_ids, b.stats_prefill_tokens, b.stats_prompt_tokens = False, None, [], 0, 0
    random.seed(3)
    prompt = "".join(random.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(1500))
    outs = []
    for chunk in (0, 64, 700, 2048):
        b.prefill_chunk = chunk
        outs.append(b._gen([prompt], 16, 4000)[0])
    assert len(set(outs)) == 1


def test_prefix_cache_reuse_gives_identical_tokens():
    """Reusing the KV cache of a shared prompt prefix (next question of the same shop, next variant, repair prompt) must not change the
    greedy output, and must really skip the shared part."""
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    import random
    from datamate.llm import HFBackend

    torch.manual_seed(1)
    cfg = tf.Qwen3Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
    model = tf.Qwen3ForCausalLM(cfg).eval()

    class BE(dict):
        def to(self, dev): return self

    class Tok:
        pad_token_id, eos_token_id = 0, 1

        def __call__(self, p, return_tensors=None, add_special_tokens=False, truncation=False, max_length=None, padding=False):
            ps = [p] if isinstance(p, str) else p
            t = torch.tensor([[2 + (ord(c) % 250) for c in x][-max_length:] for x in ps])
            return BE(input_ids=t, attention_mask=torch.ones_like(t))

        def decode(self, row, skip_special_tokens=True): return ",".join(map(str, row.tolist()))

    def make(prefix_cache):
        b = HFBackend.__new__(HFBackend)
        b.torch, b.model, b.tok, b.max_in = torch, model, Tok(), 4000
        b.prefill_chunk, b.prefix_cache, b._pc, b._pc_ids = 300, prefix_cache, None, []
        b.stats_prefill_tokens = b.stats_prompt_tokens = 0
        return b

    random.seed(5)
    rnd = lambda k: "".join(random.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(k))
    shop1, shop2 = rnd(1200), rnd(900)
    prompts = [shop1 + " Q1 variant 0", shop1 + " Q1 variant 1", shop1 + " Q1 variant 1" + rnd(200) + " repair", shop1 + " Q2 variant 0",
               shop2 + " Q3 variant 0", shop2 + " Q3 variant 0", shop1 + " Q4"]
    ref, fast = make(False), make(True)
    for pr in prompts:
        assert fast._gen([pr], 12, 4000) == ref._gen([pr], 12, 4000)
    assert fast.stats_prefill_tokens < 0.5 * fast.stats_prompt_tokens   # most of the prompt tokens were NOT recomputed


def _calib_files(tmp_path, rows):
    """rows: (confidence, is_correct) for number questions."""
    log, gold = tmp_path / "log.jsonl", tmp_path / "gold.jsonl"
    with open(log, "w") as fl, open(gold, "w") as fg:
        for i, (conf, ok) in enumerate(rows):
            fl.write(json.dumps({"id": f"q{i}", "confidence": conf, "top_candidate": {"type": "number", "answer": 1.0 if ok else 2.0}}) + "\n")
            fg.write(json.dumps({"id": f"q{i}", "type": "number", "answer": 1.0}) + "\n")
    return log, gold


def test_calibrate_keeps_defaults_on_small_or_noisy_samples(tmp_path):
    from datamate.evaluate import calibrate
    rows = [(0.6, True)] * 20 + [(0.4, False)] * 5 + [(1.0, True)] * 20      # n = 45 < 100
    log, gold = _calib_files(tmp_path, rows)
    out = calibrate(log, gold)
    assert (out["threshold_answer"], out["threshold_unanswerable"]) == (0.6, 0.8)


def test_calibrate_moves_thresholds_only_for_a_clear_gain(tmp_path):
    from datamate.evaluate import calibrate
    # 60 answers at 0.6 share that are mostly WRONG: raising threshold_answer above 0.6 avoids many -1 -> big, real gain on n=160
    rows = [(0.6, False)] * 45 + [(0.6, True)] * 15 + [(1.0, True)] * 100
    log, gold = _calib_files(tmp_path, rows)
    out = calibrate(log, gold)
    assert out["threshold_answer"] > 0.6


def test_oom_halves_the_prefill_chunk_before_cutting_context():
    from datamate.llm import HFBackend

    class T:
        def empty_cache(self): pass

    b = HFBackend.__new__(HFBackend)
    b.torch = type("X", (), {"cuda": T()})()
    b.batch_size, b.max_in, b.max_new, b.prefill_chunk, b._pc, b._pc_ids = 1, 12000, 10, 2048, None, []
    b._prompt = lambda m: m
    calls = []

    def fake_gen(prompts, mx, max_len):
        calls.append((b.prefill_chunk, max_len))
        if b.prefill_chunk > 512:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return ["ok"]

    b._gen = fake_gen
    assert b.generate_batch(["p"]) == ["ok"]
    assert calls[-1] == (512, 12000) and b.prefill_chunk == 512      # chunk halved 2048 -> 1024 -> 512, context untouched


def test_generation_stops_when_the_code_block_closes():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from datamate.llm import HFBackend

    class Tok:
        def decode(self, ids, skip_special_tokens=True): return "".join(chr(int(i)) for i in ids)

    b = HFBackend.__new__(HFBackend)
    b.torch, b.tok, b.stop_code_end = torch, Tok(), True
    prompt = "```\nprompt text with a fence```\n"
    plen = len(prompt)
    crit = b._stop_kw(plen)["stopping_criteria"][0]
    enc = lambda t: torch.tensor([[ord(c) for c in prompt + t]])
    assert not bool(crit(enc(""), None)[0])                      # fences inside the PROMPT never stop generation
    assert not bool(crit(enc("```python\nx = 1\n"), None)[0])     # an OPENING fence does not stop it
    assert not bool(crit(enc("```python\nx = 1\n``"), None)[0])
    assert bool(crit(enc("```python\nx = 1\n```\n"), None)[0])    # the closing fence does
    b.stop_code_end = False
    assert b._stop_kw(plen) == {}


def test_all_configs_load_and_vllm_keys_are_known():
    import glob
    from datamate.config import load_config
    for f in sorted(glob.glob("configs/*.json")):
        if f.endswith("model.json"):
            continue
        cfg = load_config(f)
        assert cfg["backend"] in ("hf", "vllm", "mock"), f
        if cfg["backend"] == "vllm":
            known = {"dtype", "quantization", "tensor_parallel_size", "gpu_memory_utilization", "max_model_len", "enforce_eager", "chat_template_kwargs", "max_num_seqs"}
            assert set(cfg["vllm"]) <= known, (f, set(cfg["vllm"]) - known)
            assert cfg["vllm"]["max_model_len"] > cfg["generation"]["max_new_tokens"] + 5000, f
