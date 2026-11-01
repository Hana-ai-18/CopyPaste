"""The DataMate agent: generate code variants -> execute -> repair -> vote -> answer / abstain / unanswerable."""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field

from .answer_format import answer_key, make_record
from .llm import LLMBackend
from .prompts import build_messages, build_repair_messages
from .retrieval import ExemplarBank
from .sandbox import ExecResult, LoadInfo, SandboxClient

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


def extract_code(text: str) -> str | None:
    """Last python code block of an LLM reply (handles <think> blocks and unterminated fences)."""
    text = _THINK.sub("", text or "")
    blocks = _FENCE.findall(text)
    if blocks:
        with_ans = [b for b in blocks if "answer" in b]
        return (with_ans or blocks)[-1].strip()
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", text, re.S | re.I)  # unterminated
    if m:
        return m.group(1).strip()
    if "answer_type" in text or "declare_unanswerable" in text:
        return text.strip()
    return None


@dataclass
class Candidate:
    variant: int
    code: str | None
    result: ExecResult | None
    key: str | None = None
    repairs: int = 0
    messages: list = field(default_factory=list)


@dataclass
class AgentResult:
    record: dict            # submission record WITHOUT id: {"type":..., "answer":...}
    decision: str           # answer | unanswerable | abstain
    confidence: float
    top_record: dict | None
    n_run: int
    votes: dict
    candidates: list
    elapsed: float


class BusinessContext:
    """One loaded business: profile text + sandbox state."""

    def __init__(self, business_id: str, info: LoadInfo):
        self.business_id = business_id
        self.info = info


class DataMateAgent:
    def __init__(self, llm: LLMBackend, sandbox: SandboxClient, cfg: dict, bank: ExemplarBank | None = None):
        self.llm, self.sb, self.cfg = llm, sandbox, cfg
        self.a = cfg["agent"]
        self.bank = bank or ExemplarBank([])
        self.fast_mode = False

    # ------------------------------------------------------------------ prompt
    def _messages(self, ctx: BusinessContext, rulebook: str, question: str, as_of: str, variant: int, exemplars: list[dict]):
        budgets = {"readme": self.a["max_readme_chars"], "rulebook": self.a["max_rulebook_chars"], "profile": self.a["max_profile_chars"]}
        max_in = int(self.cfg.get("hf", {}).get("max_input_tokens", 12000)) if self.cfg.get("backend") == "hf" else int(self.cfg.get("vllm", {}).get("max_model_len", 16384)) - int(self.cfg["generation"]["max_new_tokens"]) - 200
        max_in = min(max_in, 15500)
        msgs = None
        for _ in range(8):
            msgs = build_messages(question, as_of, variant, ctx.info.readme, rulebook, ctx.info.profile, exemplars, budgets)
            if self.llm.count_tokens(msgs) <= max_in:
                break
            budgets = {k: int(v * 0.75) for k, v in budgets.items()}
            if len(exemplars) > 1:
                exemplars = exemplars[:-1]
        return msgs

    # ------------------------------------------------------------------ one question
    def answer(self, ctx: BusinessContext, rulebook: str, question: str, as_of: str) -> AgentResult:
        t0 = time.time()
        stages = list(self.a["stages"])
        if self.fast_mode:
            stages = [stages[0]]
        max_repairs = 0 if self.fast_mode else int(self.a["max_repairs"])
        limit = float(self.a["per_question_time_s"])
        exemplars = self.bank.search(question, int(self.a["n_exemplars"]))
        cands: list[Candidate] = []
        done = 0
        for st in stages:
            variants = list(range(done, st))
            if not variants:
                continue
            new = self._run_variants(ctx, rulebook, question, as_of, variants, exemplars, max_repairs, t0, limit)
            cands.extend(new)
            done = st
            votes = self._votes(cands)
            if votes and len(votes) == 1 and sum(votes.values()) == done:  # unanimous -> stop early
                break
            if time.time() - t0 > limit:
                break
        return self._decide(cands, done, time.time() - t0)

    def _run_variants(self, ctx, rulebook, question, as_of, variants, exemplars, max_repairs, t0, limit) -> list[Candidate]:
        batch = [self._messages(ctx, rulebook, question, as_of, v, exemplars) for v in variants]
        texts = self.llm.generate_batch(batch)
        cands = [Candidate(variant=v, code=extract_code(t), result=None, messages=m) for v, t, m in zip(variants, texts, batch)]
        for c in cands:
            self._execute(c, as_of, self._exec_timeout(t0, limit))
        for _ in range(max_repairs):
            todo = [c for c in cands if (c.result is None or not c.result.ok)]
            if not todo or time.time() - t0 > limit:
                break
            rep = []
            for c in todo:
                err = (c.result.error if c.result else None) or "no python code block was found in your reply"
                rep.append(build_repair_messages(c.messages, c.code or "# (no code)", err, c.result.stdout if c.result else ""))
            fixed = self.llm.generate_batch(rep)
            for c, m, t in zip(todo, rep, fixed):
                new_code = extract_code(t)
                c.repairs += 1
                c.messages = m
                if new_code:
                    c.code = new_code
                    self._execute(c, as_of, self._exec_timeout(t0, limit))
        return cands

    def _exec_timeout(self, t0: float, limit: float) -> float:
        """Per-run timeout: the configured value, shrunk when the question's time budget is nearly used up."""
        left = limit - (time.time() - t0)
        return float(max(10.0, min(float(self.a["exec_timeout_s"]), left)))

    def _execute(self, c: Candidate, as_of: str, timeout: float) -> None:
        if not c.code:
            c.result, c.key = ExecResult(ok=False, error="no code"), None
            return
        c.result = self.sb.run(c.code, as_of, timeout=timeout)
        if c.result.ok and c.result.answer_type:
            try:
                c.key = answer_key(c.result.answer_type, c.result.answer)
            except Exception as e:  # malformed value that slipped through
                c.result = ExecResult(ok=False, error=f"invalid answer value: {e}", stdout=c.result.stdout)
                c.key = None
        else:
            c.key = None

    # ------------------------------------------------------------------ voting
    @staticmethod
    def _votes(cands: list[Candidate]) -> Counter:
        return Counter(c.key for c in cands if c.key)

    def _decide(self, cands: list[Candidate], n_run: int, elapsed: float) -> AgentResult:
        votes = self._votes(cands)
        log_cands = [c for c in cands]
        if not votes or n_run == 0:
            return AgentResult({"type": "abstain"}, "abstain", 0.0, None, n_run, {}, log_cands, elapsed)
        first_idx = {}
        for c in cands:
            if c.key and c.key not in first_idx:
                first_idx[c.key] = c.variant
        # most votes; ties -> candidate of the earliest variant (variant 0 = baseline prompt)
        top_key = sorted(votes, key=lambda k: (-votes[k], first_idx[k]))[0]
        top = next(c for c in cands if c.key == top_key)
        conf = votes[top_key] / n_run
        r = top.result
        if r.answer_type == "unanswerable":
            top_rec = {"type": "unanswerable", "reason": r.reason or ""}
            ok = conf >= float(self.a["threshold_unanswerable"])
            rec, decision = (top_rec, "unanswerable") if ok else ({"type": "abstain"}, "abstain")
        else:
            top_rec = {"type": r.answer_type, "answer": r.answer}
            ok = conf >= float(self.a["threshold_answer"])
            rec, decision = (top_rec, "answer") if ok else ({"type": "abstain"}, "abstain")
        return AgentResult(rec, decision, conf, top_rec, n_run, dict(votes), log_cands, elapsed)


def to_submission(qid: str, rec: dict) -> dict:
    t = rec["type"]
    if t == "abstain":
        return make_record(qid, "abstain")
    if t == "unanswerable":
        return make_record(qid, "unanswerable", reason=rec.get("reason"))
    return make_record(qid, t, rec["answer"])
