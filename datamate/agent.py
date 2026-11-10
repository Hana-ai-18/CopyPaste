"""The DataMate agent: generate code variants -> execute -> repair -> vote -> answer / abstain / unanswerable."""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field

from .answer_format import answer_key, make_record
from .llm import LLMBackend
from .qnorm import Vocab, amounts_note, default_vocab, normalize_question
from .rulebook import select_rulebook
from .intent import audit_code, resolve_periods, rule_checklist
from .prompts import build_audit_messages, build_empty_repair_messages, build_messages, build_repair_messages
from .retrieval import ExemplarBank
from .sandbox import ExecResult, LoadInfo, SandboxClient

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


def empty_reason(atype: str | None, answer) -> str | None:
    """Describe why an executed answer looks like a wiped-out filter (0 / empty list / empty table), else None."""
    try:
        if atype in ("number", "percent") and float(answer) == 0:
            return "the answer is 0"
        if atype == "list" and len(answer) == 0:
            return "the list is empty"
        if atype == "table" and len(answer.get("rows", [])) == 0:
            return "the table has no rows"
        if atype == "string" and not str(answer).strip():
            return "the string is empty"
    except Exception:
        return None
    return None


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
    issues: list = field(default_factory=list)      # self-check findings that survived the audit round
    question: str = ""


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
    ref: dict | None = None      # independent second computation (refcalc.py): {'answer':..., 'agree': bool} or None when it had no opinion


class BusinessContext:
    """One loaded business: profile text + sandbox state."""

    def __init__(self, business_id: str, info: LoadInfo):
        self.business_id = business_id
        self.evidence = ""            # README-independent text about this shop (its questions, table/column names): used to pick rulebook rows
        self.full_rulebook = ""       # the whole split rulebook, re-selected with a smaller budget when the prompt is too long
        self.info = info
        self._vocab = None

    def vocab(self):
        """Default Vietnamese vocabulary extended with this shop's own words (README, profile): brands/product names are then never 'repaired'."""
        if self._vocab is None:
            import copy

            v = copy.deepcopy(default_vocab())
            try:
                v.add_text(self.info.readme + "\n" + self.info.profile)
            except Exception:  # noqa: BLE001
                pass
            self._vocab = v
        return self._vocab

    def lexicon(self):
        if getattr(self, '_lex', None) is None:
            import copy

            from .slang import default_lexicon

            lx = copy.deepcopy(default_lexicon())
            try:
                lx.add_text(self.info.readme + "\n" + self.info.profile)
            except Exception:  # noqa: BLE001
                pass
            self._lex = lx
        return self._lex


class DataMateAgent:
    def __init__(self, llm: LLMBackend, sandbox: SandboxClient, cfg: dict, bank: ExemplarBank | None = None):
        self.llm, self.sb, self.cfg = llm, sandbox, cfg
        self.a = cfg["agent"]
        self.bank = bank or ExemplarBank([])
        self.fast_mode = False
        self.pace_s = None

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
            # shrink the README / profile / examples first; the rulebook is the source of truth for the answers and is re-selected
            # (whole rows dropped, never cut mid-text) only when that is not enough
            budgets["readme"] = max(800, int(budgets["readme"] * 0.7))
            budgets["profile"] = max(1800, int(budgets["profile"] * 0.8))
            if len(exemplars) > 1:
                exemplars = exemplars[:-1]
            if budgets["readme"] <= 800 and budgets["profile"] <= 1800 and len(exemplars) <= 1 and ctx.full_rulebook:
                budgets["rulebook"] = max(6000, int(budgets["rulebook"] * 0.85))
                rulebook = select_rulebook(ctx.full_rulebook, ctx.info.readme, budgets["rulebook"], ctx.evidence)
        return msgs

    # ------------------------------------------------------------------ one question
    def answer(self, ctx: BusinessContext, rulebook: str, question: str, as_of: str) -> AgentResult:
        t0 = time.time()
        stages = list(self.a["stages"])
        if self.fast_mode:
            stages = [stages[0]]
        max_repairs = 0 if self.fast_mode else int(self.a["max_repairs"])
        limit = float(self.a["per_question_time_s"])
        if getattr(self, "pace_s", None):
            limit = min(limit, float(self.pace_s))
        try:
            qn = normalize_question(question, ctx.vocab(), lex=ctx.lexicon())
        except Exception:  # noqa: BLE001 - normalisation is a help, never a failure point
            qn = question
        if self.a.get("engine_first", True) and not self.fast_mode:
            # Plain questions of the engine's families need no model at all: faster (inference time breaks leaderboard ties) and not exposed to sampling noise.
            try:
                ref = self.sb.ref(qn, as_of, timeout=60.0)
            except Exception:  # noqa: BLE001
                ref = None
            if ref:
                rec = {"type": ref[0], "answer": ref[1]}
                res = AgentResult(rec, "answer", 1.0, rec, 0, {}, [], time.time() - t0)
                res.ref = {"answer": ref[1], "agree": None, "used": True, "engine_first": True}
                return res
        exemplars = self.bank.search(qn, int(self.a["n_exemplars"]))
        try:
            amt = amounts_note(question)
        except Exception:  # noqa: BLE001
            amt = ""
        if qn.strip() != question.strip():
            question = f"{question.strip()}\n(Reading with shorthand and typing artefacts expanded automatically - may be imperfect, the original above is authoritative: {qn.strip()})"
        if amt:
            question = f"{question.strip()}\n(Amounts in the question: {amt})"
        try:
            per = resolve_periods(qn, as_of)
            if per:
                question = f"{question.strip()}\n(Periods computed in Python from as_of={as_of} with the rulebook's rules - use these dates exactly:\n" + "\n".join(per) + ")"
            chk = rule_checklist(qn)
            if chk:
                question = f"{question.strip()}\n(Rules that apply to this question - the rulebook above has the details:\n" + "\n".join("* " + c for c in chk) + ")"
        except Exception:  # noqa: BLE001 - advisory only
            pass
        self._q_for_audit = qn
        try:
            from .intent import line_columns_from_profile

            self._line_cols = line_columns_from_profile(ctx.info.profile)
        except Exception:  # noqa: BLE001
            self._line_cols = None
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
        res = self._decide(cands, done, time.time() - t0)
        return self._crosscheck(res, qn, as_of)

    def _crosscheck(self, res: AgentResult, question: str, as_of: str) -> AgentResult:
        """Independent engine (refcalc) for plain 'revenue / number of orders in one period' questions (99% right on the train shops, see CA_KHO_PRIVATE_TEST D4).
        Agreement: keep. Disagreement or model abstained: the ENGINE's value is submitted (three identical wrong programs still agree with each other,
        so the vote alone cannot be trusted; a blank earns 0, the engine earns about +1 in expectation). No opinion from the engine: no change."""
        if self.fast_mode or res.decision == "unanswerable" or (res.record.get("type") not in ("number", "percent", "abstain")):
            return res
        try:
            ref = self.sb.ref(question, as_of, timeout=60.0)
        except Exception:  # noqa: BLE001
            ref = None
        if not ref:
            return res
        try:
            tol = 0.5 if ref[0] == "number" else 0.0051
            same = res.record.get("type") == ref[0] and abs(float(ref[1]) - float(res.record["answer"])) < tol
        except Exception:  # noqa: BLE001
            return res
        res.ref = {"answer": ref[1], "agree": bool(same), "used": not same}
        if not same:
            rec = {"type": ref[0], "answer": ref[1]}
            res.record, res.top_record, res.decision = rec, rec, "answer"
            res.confidence = max(res.confidence, 0.6)
        return res

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
        if max_repairs > 0 and time.time() - t0 <= limit:
            self._empty_round(cands, as_of, t0, limit)
        if max_repairs > 0 and time.time() - t0 <= limit:
            self._audit_round(cands, as_of, t0, limit)
        else:
            self._audit_mark(cands)
        return cands

    def _audit_mark(self, cands: list[Candidate]) -> None:
        for c in cands:
            if c.result is not None and c.result.ok and c.code:
                try:
                    c.issues = audit_code(getattr(self, "_q_for_audit", ""), c.code, c.result.answer_type, c.result.answer, getattr(self, "_line_cols", None), getattr(c.result, "lineage", None))
                except Exception:  # noqa: BLE001
                    c.issues = []

    def _audit_round(self, cands: list[Candidate], as_of: str, t0: float, limit: float) -> None:
        """One targeted repair for candidates whose code breaks an unmistakable rule (e.g. revenue without a status filter). The old result is kept if the retry fails."""
        self._audit_mark(cands)
        todo = [c for c in cands if c.issues and c.code]
        if not todo:
            return
        rep = [build_audit_messages(c.messages, c.code, c.issues, c.result.stdout) for c in todo]
        fixed = self.llm.generate_batch(rep)
        for c, m, t in zip(todo, rep, fixed):
            new_code = extract_code(t)
            if not new_code or new_code.strip() == (c.code or "").strip():
                continue
            old = (c.code, c.result, c.key, c.messages, c.issues)
            c.code, c.messages = new_code, m
            self._execute(c, as_of, self._exec_timeout(t0, limit))
            c.repairs += 1
            if c.result and c.result.ok and c.key:
                try:
                    c.issues = audit_code(getattr(self, "_q_for_audit", ""), c.code, c.result.answer_type, c.result.answer, getattr(self, "_line_cols", None), getattr(c.result, "lineage", None))
                except Exception:  # noqa: BLE001
                    c.issues = []
            else:
                c.code, c.result, c.key, c.messages, c.issues = old

    def _empty_round(self, cands: list[Candidate], as_of: str, t0: float, limit: float) -> None:
        """One extra round for candidates that ran fine but returned 0 / empty: usually a wrong filter. Keep the old result unless the retry is valid."""
        todo = []
        for c in cands:
            if c.result is not None and c.result.ok and c.code:
                why = empty_reason(c.result.answer_type, c.result.answer)
                if why:
                    todo.append((c, why))
        if not todo:
            return
        rep = [build_empty_repair_messages(c.messages, c.code, why, c.result.stdout) for c, why in todo]
        fixed = self.llm.generate_batch(rep)
        for (c, _), m, t in zip(todo, rep, fixed):
            new_code = extract_code(t)
            if not new_code:
                continue
            old = (c.code, c.result, c.key, c.messages)
            c.code, c.messages = new_code, m
            self._execute(c, as_of, self._exec_timeout(t0, limit))
            c.repairs += 1
            if not (c.result and c.result.ok and c.key):
                c.code, c.result, c.key, c.messages = old

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
        flagged = [c for c in cands if c.key == top_key and c.issues]
        if flagged and len(flagged) == sum(1 for c in cands if c.key == top_key):
            conf *= 0.75           # every voter for this answer still breaks a rule the self-check is sure about: do not trust unanimity
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
