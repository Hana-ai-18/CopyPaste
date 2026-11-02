"""Entry point: `python -m datamate.run --data_dir /data --out /result/submission.jsonl --logs /result/logs`.

Reads  <data_dir>/questions.jsonl, rulebook.md, calendar.csv, data/<business_id>/...
Writes one JSON line per question (type: number|percent|string|date|boolean|list|table|unanswerable|abstain),
plus a per-question log (id, final answer, elapsed time, generated code, execution output).
Questions that are not answered (error / time budget) are written as `abstain`.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path

T0 = time.time()


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _find_business_dir(data_dir: Path, biz: str) -> Path | None:
    for p in (data_dir / "data" / biz, data_dir / biz):
        if p.is_dir():
            return p
    return None


def _cand_log(c, max_code: int = 6000) -> dict:
    r = c.result
    return {
        "variant": c.variant, "repairs": c.repairs, "ok": bool(r and r.ok),
        "answer_type": r.answer_type if r else None, "answer": r.answer if r else None,
        "error": (r.error if r else "no result"), "stdout": (r.stdout[:600] if r else ""),
        "exec_s": r.elapsed if r else 0.0, "key": c.key, "code": (c.code or "")[:max_code],
    }


def resource_stats() -> dict:
    """Peak GPU memory (this process) and peak host RAM - read by datamate.preflight to size the real run."""
    out: dict = {}
    try:
        import resource

        out["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)      # KB on Linux
    except Exception:  # noqa: BLE001
        pass
    try:
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(0) / 1e9, 2)
            out["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except Exception:  # noqa: BLE001
        pass
    return out


def shard_questions(questions: list[dict], index: int, n: int) -> list[dict]:
    """Deterministic, balanced split by business (largest business first -> lightest shard). Question order is kept."""
    counts: dict[str, int] = {}
    for q in questions:
        counts[q["business_id"]] = counts.get(q["business_id"], 0) + 1
    load = [0] * n
    owner: dict[str, int] = {}
    for biz in sorted(counts, key=lambda b: (-counts[b], b)):
        k = min(range(n), key=lambda i: (load[i], i))
        owner[biz] = k
        load[k] += counts[biz]
    return [q for q in questions if owner[q["business_id"]] == index]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DataMate - Zalo AI Challenge 2026")
    ap.add_argument("--data_dir", default="/data", help="split folder (questions.jsonl, rulebook.md, calendar.csv, data/)")
    ap.add_argument("--out", default="/result/submission.jsonl")
    ap.add_argument("--logs", default="/result/logs")
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--set", action="append", default=[], help="override config, e.g. --set agent.stages=[3,5]")
    ap.add_argument("--questions", default=None, help="override path of questions.jsonl")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N questions (use with --limit for held-out slices of train)")
    ap.add_argument("--limit", type=int, default=None, help="only N questions after --offset (debug / dev slices)")
    ap.add_argument("--business", default=None, help="only this business id (debug)")
    ap.add_argument("--biz_fold", default=None, help="K/N: only the businesses whose sorted index %% N == K (a hold-out of SHOPS, to measure generalisation to unseen shops)")
    ap.add_argument("--exclude_fold", default=None, help="K/N: drop those businesses (use for building exemplars / training signals that must not see the hold-out shops)")
    ap.add_argument("--shard", default=None, help="I/N: only the businesses assigned to shard I of N (balanced by question count) - one process per GPU")
    ap.add_argument("--resume", action="store_true", help="keep answers already in --out and only answer the missing ones")
    ap.add_argument("--eval", action="store_true", help="after the run, score against <data_dir>/answers.jsonl if it exists (training data)")
    args = ap.parse_args(argv)

    from .agent import BusinessContext, DataMateAgent, to_submission
    from .config import load_config
    from .llm import build_backend, set_all_seeds
    from .retrieval import ExemplarBank
    from .rulebook import select_rulebook
    from .sandbox import SandboxClient

    cfg = load_config(args.config, args.set)
    seed = int(cfg["seed"])
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    data_dir = Path(args.data_dir)
    out_path, log_dir = Path(args.out), Path(args.logs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    qpath = Path(args.questions) if args.questions else data_dir / "questions.jsonl"
    questions = _read_jsonl(qpath)
    if args.business:
        wanted = {b.strip() for b in args.business.split(",")}
        questions = [q for q in questions if q["business_id"] in wanted]
    for flag, keep in ((args.biz_fold, True), (args.exclude_fold, False)):
        if flag:
            fk, fn = (int(x) for x in flag.split("/"))
            allb = sorted({q["business_id"] for q in questions})
            sel = {b for i, b in enumerate(allb) if i % fn == fk}
            questions = [q for q in questions if (q["business_id"] in sel) == keep]
    if args.offset or args.limit:
        questions = questions[args.offset: (args.offset + args.limit) if args.limit else None]
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        questions = shard_questions(questions, si, sn)
    order = [q["id"] for q in questions]
    rulebook_path = data_dir / "rulebook.md"
    calendar_path = data_dir / "calendar.csv"
    print(f"[run] {len(questions)} questions | rulebook={rulebook_path.exists()} calendar={calendar_path.exists()}", flush=True)
    rulebook = rulebook_path.read_text(encoding="utf-8-sig", errors="replace") if rulebook_path.exists() else ""

    records: dict[str, dict] = {}
    if args.resume and out_path.exists():
        for r in _read_jsonl(out_path):
            records[r["id"]] = r
        # questions abstained only because the time budget ended (or a transient CUDA error) must be retried, not kept
        try:
            for i in json.load(open(log_dir / "unfinished.json", encoding="utf-8")):
                records.pop(i, None)
        except (OSError, ValueError):
            pass
        print(f"[run] resume: {len(records)} answers already present", flush=True)

    # compliance guard: total parameters of the model must not exceed the limit (DataMate: 15B, MoE counts all parameters)
    model_info = None
    if cfg["backend"] in ("hf", "vllm") and Path(cfg["model_path"]).is_dir():
        from .model_check import count_params

        model_info = count_params(cfg["model_path"])
        print(f"[run] model {cfg['model_path']}: {model_info['params_b']}B parameters ({model_info['n_files']} safetensors files)", flush=True)
        if model_info["params"] / 1e9 > float(cfg.get("max_model_params_b", 15)):
            raise SystemExit(f"STOP: the model has {model_info['params_b']}B parameters, above the allowed {cfg.get('max_model_params_b', 15)}B")

    # IMPORTANT: start the sandbox worker BEFORE the GPU model is loaded (no fork of a CUDA process).
    sb = SandboxClient()
    t_model = time.time()
    set_all_seeds(seed)
    llm = build_backend(cfg)
    model_load_s = round(time.time() - t_model, 1)
    print(f"[run] backend={llm.name} loaded in {model_load_s}s", flush=True)
    bank = ExemplarBank.load(cfg["agent"].get("exemplars_path"), cfg["agent"].get("seed_exemplars_path", "exemplars/seed_exemplars.jsonl"))
    print(f"[run] exemplar bank: {len(bank)} items", flush=True)
    agent = DataMateAgent(llm, sb, cfg, bank)

    mode = "a" if args.resume else "w"
    fout = open(out_path, mode, encoding="utf-8")
    flog = open(log_dir / "log.jsonl", mode, encoding="utf-8")
    json.dump({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version, "platform": platform.platform(),
               "config": cfg, "model": model_info, "n_questions": len(questions)}, open(log_dir / "run_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)

    todo = [q for q in questions if q["id"] not in records]
    by_biz: dict[str, list[dict]] = {}
    for q in todo:
        by_biz.setdefault(q["business_id"], []).append(q)
    unfinished: set[str] = set()
    budget = float(cfg["time_budget_s"])
    remaining = len(todo)
    n_done = 0

    def emit(q: dict, rec: dict, extra: dict) -> None:
        records[q["id"]] = rec
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        flog.write(json.dumps({"id": q["id"], "business_id": q["business_id"], "question": q["question"], "as_of": q.get("as_of_date"),
                               "submitted": rec, **extra}, ensure_ascii=False, default=str) + "\n")
        flog.flush()

    for biz in sorted(by_biz):
        qs = by_biz[biz]
        bdir = _find_business_dir(data_dir, biz)
        ctx = None
        rb_biz = rulebook
        if bdir is None:
            err = f"business dir for {biz} not found"
        else:
            try:
                t = time.time()
                info = sb.load(bdir, biz, rulebook_path if rulebook_path.exists() else None,
                               calendar_path if calendar_path.exists() else None, cfg["agent"]["max_profile_chars"])
                ctx = BusinessContext(biz, info)
                from .qnorm import normalize_question

                evidence = " ".join(normalize_question(x["question"]) for x in qs) + " " + " ".join(
                    str(t) + " " + " ".join(map(str, cols if isinstance(cols, (list, tuple)) else getattr(cols, "columns", []))) for t, cols in info.tables.items())
                ctx.evidence, ctx.full_rulebook = evidence, rulebook
                rb_biz = select_rulebook(rulebook, info.readme, int(cfg["agent"]["max_rulebook_chars"]), evidence)
                err = None
                print(f"[run] {biz}: {len(info.tables)} tables loaded in {time.time() - t:.1f}s warnings={info.warnings[:2]}", flush=True)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
        for q in qs:
            tq = time.time()
            elapsed_total = tq - T0
            if err is not None or elapsed_total > budget:
                why = err or "global time budget exhausted"
                emit(q, {"id": q["id"], "type": "abstain"}, {"decision": "abstain", "error": why, "elapsed_s": 0.0})
                unfinished.add(q["id"])
                remaining -= 1
                continue
            allow = (budget - elapsed_total) / max(remaining, 1)
            agent.fast_mode = allow < float(cfg["agent"]["min_time_per_question_s"])
            agent.pace_s = max(float(cfg["agent"]["min_time_per_question_s"]), 1.6 * allow)       # pace: one slow question cannot eat the budget of the rest
            try:
                res = agent.answer(ctx, rb_biz, q["question"], q["as_of_date"])
                rec = to_submission(q["id"], res.record)
                extra = {"decision": res.decision, "confidence": round(res.confidence, 3), "n_run": res.n_run, "votes": res.votes,
                         "top_candidate": res.top_record, "elapsed_s": round(time.time() - tq, 2), "fast_mode": agent.fast_mode, "ref": getattr(res, "ref", None),
                         "candidates": [_cand_log(c) for c in res.candidates]}
            except Exception:  # noqa: BLE001 - one bad question must never kill the run
                rec = {"id": q["id"], "type": "abstain"}
                extra = {"decision": "abstain", "error": traceback.format_exc()[-1500:], "elapsed_s": round(time.time() - tq, 2)}
                if "out of memory" in extra["error"].lower():
                    unfinished.add(q["id"])
            emit(q, rec, extra)
            remaining -= 1
            n_done += 1
            if n_done % 10 == 0 or n_done == len(todo):
                print(f"[run] {n_done}/{len(todo)} answered | elapsed {time.time() - T0:.0f}s | last={rec['type']} ({extra['decision']})", flush=True)

    fout.close()
    flog.close()
    json.dump(sorted(unfinished), open(log_dir / "unfinished.json", "w", encoding="utf-8"))
    if unfinished:
        print(f"[run] {len(unfinished)} questions NOT answered (time budget / CUDA OOM) -> written as abstain; run again with --resume to answer them", flush=True)
    # final file in question order, every id present (missing -> abstain)
    final = [records.get(i, {"id": i, "type": "abstain"}) for i in order]
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    try:
        info = json.load(open(log_dir / "run_info.json", encoding="utf-8"))
        info.update({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "total_seconds": round(time.time() - T0, 1), "n_written": len(final),
                     "model_load_s": model_load_s, **resource_stats(),
                     "prompt_tokens": getattr(llm, "stats_prompt_tokens", None), "prefill_tokens_computed": getattr(llm, "stats_prefill_tokens", None)})
        json.dump(info, open(log_dir / "run_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    except Exception:
        pass
    print(f"[run] done: {len(final)} lines -> {out_path} in {time.time() - T0:.0f}s", flush=True)

    if args.eval and (data_dir / "answers.jsonl").exists():
        from .evaluate import report

        report(out_path, data_dir / "answers.jsonl", questions_ids=set(order))
    llm.close()
    sb.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
