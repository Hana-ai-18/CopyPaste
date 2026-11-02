"""LLM backends. Everything runs LOCALLY from weights on disk (no network, no API).

* HFBackend    - transformers (+ bitsandbytes 4-bit). Easiest, works on a Kaggle T4.
* VLLMBackend  - vLLM offline engine. Fastest on the scoring machine (2x RTX 5090).
* MockBackend  - scripted responses, used for unit tests / smoke tests only (never in predict.sh).
All backends decode greedily (deterministic).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

Messages = "list[dict]"


class LLMBackend:
    name = "base"

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        raise NotImplementedError

    def generate_batch(self, batch: list[list[dict]], max_new_tokens: int | None = None) -> list[str]:
        return [self.generate(m, max_new_tokens) for m in batch]

    def count_tokens(self, messages: list[dict]) -> int:
        return sum(len(m["content"]) for m in messages) // 3 + 8 * len(messages)

    def close(self) -> None:
        pass


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- transformers
class HFBackend(LLMBackend):
    name = "hf"

    def __init__(self, model_path: str, cfg: dict, gen_cfg: dict, seed: int = 0):
        import torch
        import transformers
        from packaging import version
        from transformers import AutoModelForCausalLM, AutoTokenizer

        set_all_seeds(seed)
        self.torch = torch
        self.cfg = cfg
        self.max_new = int(gen_cfg.get("max_new_tokens", 900))
        self.ctk = cfg.get("chat_template_kwargs") or {}
        self.batch_size = max(1, int(cfg.get("batch_size", 1)))
        self.max_in = int(cfg.get("max_input_tokens", 12000))
        self.prefill_chunk = int(cfg.get("prefill_chunk", 2048))   # 0 = old behaviour (whole prompt in one forward pass)
        self.prefix_cache = bool(cfg.get("prefix_cache", True))     # keep the KV cache of the shared prompt prefix between calls
        self._pc, self._pc_ids = None, []
        self.stats_prefill_tokens = self.stats_prompt_tokens = 0
        trc = bool(cfg.get("trust_remote_code", False))
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trc)
        self.tok.padding_side = "left"
        self.tok.truncation_side = "left"  # if a prompt is ever too long, drop the START (context), never the question/generation prompt
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = getattr(torch, cfg.get("dtype", "float16"))
        # one process = one GPU: pin the whole model to it (device_map="auto" could silently spill layers to the CPU -> slow + RAM blow-up);
        # low_cpu_mem_usage streams the checkpoint shard by shard, so peak host RAM stays around one shard (~4 GB), not the model size
        kw: dict = {"device_map": {"": 0} if torch.cuda.is_available() else "auto", "trust_remote_code": trc, "low_cpu_mem_usage": True}
        new_api = version.parse(transformers.__version__) >= version.parse("4.56")
        kw["dtype" if new_api else "torch_dtype"] = dtype
        if cfg.get("load_in_4bit") and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
        self.model.eval()

    def _prompt(self, messages: list[dict]) -> str:
        try:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **self.ctk)
        except TypeError:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(self, messages: list[dict]) -> int:
        return len(self.tok(self._prompt(messages), add_special_tokens=False)["input_ids"])

    def generate_batch(self, batch, max_new_tokens=None):
        torch = self.torch
        outs: list[str] = []
        mx = int(max_new_tokens or self.max_new)
        for i in range(0, len(batch), self.batch_size):
            prompts = [self._prompt(m) for m in batch[i:i + self.batch_size]]
            try:
                outs.extend(self._gen(prompts, mx, self.max_in))
            except Exception as e:  # noqa: BLE001
                if "out of memory" not in str(e).lower():
                    raise
                # CUDA OOM (long prompt / big batch): free the cache, retry the prompts one by one with a 25 % shorter context
                self._pc, self._pc_ids = None, []
                self.torch.cuda.empty_cache()
                for pr in prompts:
                    outs.extend(self._gen([pr], mx, int(self.max_in * 0.75)))
        return outs

    def _gen(self, prompts: list[str], mx: int, max_len: int) -> list[str]:
        if self.prefill_chunk > 0:
            return [self._gen_chunked(p, mx, max_len) for p in prompts]
        torch = self.torch
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                       truncation=True, max_length=max_len).to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=mx, do_sample=False, temperature=None, top_p=None, top_k=None,
                                      pad_token_id=self.tok.pad_token_id)
        res = [self.tok.decode(row, skip_special_tokens=True) for row in gen[:, enc["input_ids"].shape[1]:]]
        del enc, gen
        return res

    def _gen_chunked(self, prompt: str, mx: int, max_len: int) -> str:
        """One prompt at a time. The prompt is processed in chunks of `prefill_chunk` tokens that fill the KV cache, then generate()
        only has to feed the last token (attention memory = chunk x length instead of length x length: on a T4 a whole 13k-token
        prefill materialised a 10 GB attention matrix and ran out of memory on every question).

        PREFIX REUSE: the KV cache is kept between calls. Every prompt of a shop starts with the same ~12k tokens (system prompt,
        README, rulebook, table profile); the 3-5 variants of a question and the repair prompts also share everything but the end.
        So for the next prompt only the tokens after the longest common prefix are processed - the big prefill is paid once per
        shop instead of once per generation."""
        torch = self.torch
        from transformers import DynamicCache

        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"].to(self.model.device)
        seq = ids[0].tolist()
        n = len(seq)
        keep = 0
        if self.prefix_cache and self._pc is not None:
            old = self._pc_ids
            lim = min(len(old), n - 1)
            while keep < lim and old[keep] == seq[keep]:
                keep += 1
            try:
                cur = int(self._pc.get_seq_length())
                if keep > 0 and cur > keep:
                    self._pc.crop(-(cur - keep))  # drop what was generated / the part that differs (negative = "remove N tokens": works in every transformers version)
                if keep > 0 and int(self._pc.get_seq_length()) != keep:
                    keep = 0
            except Exception:  # noqa: BLE001
                keep = 0
        cache = self._pc if keep > 0 else DynamicCache()
        self._pc, self._pc_ids = None, []        # (re)registered only after a successful run: an OOM must not leave a half-filled cache behind
        self.stats_prefill_tokens += (n - 1) - keep
        self.stats_prompt_tokens += n - 1
        with torch.no_grad():
            for s in range(keep, n - 1, self.prefill_chunk):
                e = min(s + self.prefill_chunk, n - 1)
                kw = {"input_ids": ids[:, s:e], "past_key_values": cache, "use_cache": True,
                      "cache_position": torch.arange(s, e, device=ids.device)}
                try:
                    self.model(**kw, logits_to_keep=1)      # no need for logits of the prefix (saves chunk x 152k vocab floats)
                except TypeError:
                    self.model(**kw)
            gen = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=cache, max_new_tokens=mx,
                                      do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id)
        out = self.tok.decode(gen[0, n:], skip_special_tokens=True)
        if self.prefix_cache:
            self._pc, self._pc_ids = cache, seq[: n - 1]
        del ids, gen
        return out

    def generate(self, messages, max_new_tokens=None):
        return self.generate_batch([messages], max_new_tokens)[0]

    def close(self):
        try:
            del self.model
            self.torch.cuda.empty_cache()
        except Exception:
            pass


# --------------------------------------------------------------------------- vLLM
class VLLMBackend(LLMBackend):
    name = "vllm"

    def __init__(self, model_path: str, cfg: dict, gen_cfg: dict, seed: int = 0):
        from vllm import LLM, SamplingParams

        set_all_seeds(seed)
        self.SamplingParams = SamplingParams
        self.max_new = int(gen_cfg.get("max_new_tokens", 900))
        self.ctk = cfg.get("chat_template_kwargs") or {}
        kw = {"model": model_path, "dtype": cfg.get("dtype", "auto"), "seed": seed,
              "tensor_parallel_size": int(cfg.get("tensor_parallel_size", 1)),
              "gpu_memory_utilization": float(cfg.get("gpu_memory_utilization", 0.88)),
              "max_model_len": int(cfg.get("max_model_len", 16384)),
              "enforce_eager": bool(cfg.get("enforce_eager", False)), "enable_prefix_caching": True}
        if cfg.get("quantization"):
            kw["quantization"] = cfg["quantization"]
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()

    def _prompt(self, messages):
        try:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **self.ctk)
        except TypeError:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(self, messages):
        return len(self.tok(self._prompt(messages), add_special_tokens=False)["input_ids"])

    def generate_batch(self, batch, max_new_tokens=None):
        sp = self.SamplingParams(temperature=0.0, max_tokens=int(max_new_tokens or self.max_new), seed=0)
        outs = self.llm.generate([self._prompt(m) for m in batch], sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def generate(self, messages, max_new_tokens=None):
        return self.generate_batch([messages], max_new_tokens)[0]


# --------------------------------------------------------------------------- mock (tests only)
class MockBackend(LLMBackend):
    """`responder(messages) -> str`. For unit tests and notebook smoke tests only."""

    name = "mock"

    def __init__(self, responder: Callable[[list[dict]], str]):
        self.responder = responder
        self.calls = 0

    def generate(self, messages, max_new_tokens=None):
        self.calls += 1
        return self.responder(messages)


class FileMockBackend(MockBackend):
    """Responses from a JSON file produced by scripts/make_demo_data.py.

    {question_text: {"variants": [code, ...], "repair": code}}; the variant index is read from
    the 'Strategy variant: k' line of the prompt. Used ONLY for the offline smoke test.
    """

    def __init__(self, path: str | Path):
        import re

        self._rx_v = re.compile(r"Strategy variant: (\d+)")
        self._rx_q = re.compile(r"Question: (.*)")
        self.table = json.loads(Path(path).read_text(encoding="utf-8"))
        super().__init__(self._respond)

    def _respond(self, messages):
        first = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        qs = self._rx_q.findall(first)  # the real question is the last "Question:" line
        v = self._rx_v.search(first)
        ent = self.table.get(qs[-1].strip() if qs else "")
        if ent is None:
            return "```python\ndeclare_unanswerable('no mock response')\n```"
        if len(messages) > 2:  # repair turn
            return "```python\n" + ent.get("repair", ent["variants"][0]) + "\n```"
        idx = int(v.group(1)) if v else 0
        return "```python\n" + ent["variants"][idx % len(ent["variants"])] + "\n```"


def build_backend(cfg: dict) -> LLMBackend:
    kind = cfg.get("backend", "hf")
    seed = int(cfg.get("seed", 0))
    path = cfg.get("model_path", "models/llm")
    if kind == "hf":
        return HFBackend(path, cfg.get("hf", {}), cfg.get("generation", {}), seed)
    if kind == "vllm":
        return VLLMBackend(path, cfg.get("vllm", {}), cfg.get("generation", {}), seed)
    if kind == "mock":
        mp = cfg.get("mock_responses")
        if not mp:
            raise ValueError("backend=mock needs --set mock_responses=<path to mock_responses.json>")
        return FileMockBackend(mp)
    raise ValueError(f"unknown backend {kind!r}")
