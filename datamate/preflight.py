"""Preflight: run a handful of REAL questions on the real GPU, then turn the measurements into settings for the full run
(instead of trusting estimates). Pure logic here; the notebook mode `preflight` does the measuring.

    python -m datamate.preflight --logs out/logs --n_questions 1000 --procs 2 --budget_h 10 --out auto_tune.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def recommend(log_rows: list[dict], run_info: dict, n_questions: int, procs: int, budget_s: float, vram_total_gb: float | None = None) -> dict:
    """-> {"set": [...cli overrides...], "advice": [...], "sec_per_question": x, "projected_hours": h}"""
    advice: list[str] = []
    sets: list[str] = []
    secs = [r["elapsed_s"] for r in log_rows if isinstance(r.get("elapsed_s"), (int, float)) and r.get("decision") is not None and not r.get("error")]
    if not secs:
        return {"set": [], "advice": ["No timed question in the preflight log - the model probably failed to run; read the log above."],
                "sec_per_question": None, "projected_hours": None}
    spq = statistics.mean(secs)
    load_s = float(run_info.get("model_load_s") or 0)
    peak = run_info.get("peak_vram_gb")
    total = vram_total_gb or run_info.get("gpu_total_gb")
    errs = sum(1 for r in log_rows if "out of memory" in str(r.get("error", "")).lower())
    if errs:
        advice.append(f"{errs} question(s) hit CUDA out-of-memory -> lowering the context; if it persists switch to the 8B model.")
        sets += ["hf.max_input_tokens=9000", "hf.batch_size=1"]
    elif peak and total and peak / total > 0.93:
        advice.append(f"VRAM peak {peak} of {total} GB is very close to the limit -> lowering max_input_tokens to stay safe.")
        sets += ["hf.max_input_tokens=9000", "hf.batch_size=1"]
    elif peak and total:
        advice.append(f"VRAM peak {peak} of {total} GB: OK.")

    def hours(s_per_q: float) -> float:
        return (load_s + n_questions * s_per_q / max(1, procs)) / 3600

    proj = hours(spq)
    usable_h = budget_s / 3600
    stages_full = proj
    if proj > usable_h:
        # 3-sample voting only (no 5-sample second stage) costs about 60 % of the full voting; single sample about 25 %
        proj3, proj1 = hours(spq * 0.6), hours(spq * 0.25)
        if proj3 <= usable_h * 0.95:
            sets.append("agent.stages=[3]")
            advice.append(f"Projected {proj:.1f} h > budget {usable_h:.1f} h with stages [3,5] -> using stages [3] (~{proj3:.1f} h).")
            proj = proj3
        elif proj1 <= usable_h * 0.95:
            sets.append("agent.stages=[1]")
            advice.append(f"Projected {proj:.1f} h > budget {usable_h:.1f} h -> single sample only (~{proj1:.1f} h): accuracy drops; prefer the 8B model or more sessions.")
            proj = proj1
        else:
            advice.append(f"Even one sample per question needs ~{proj1:.1f} h > {usable_h:.1f} h: use the 8B model, or split the run over several sessions (RESUME).")
            proj = proj1
    else:
        advice.append(f"Projected {proj:.1f} h <= budget {usable_h:.1f} h with the full voting: keep stages [3,5].")
    return {"set": sets, "advice": advice, "sec_per_question": round(spq, 1), "projected_hours": round(proj, 2), "stages_full_hours": round(stages_full, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="logs folder of the preflight run (log.jsonl, run_info.json)")
    ap.add_argument("--n_questions", type=int, required=True)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--budget_h", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logs = Path(a.logs)
    rows = [json.loads(l) for l in open(logs / "log.jsonl", encoding="utf-8") if l.strip()]
    info = json.load(open(logs / "run_info.json", encoding="utf-8"))
    rec = recommend(rows, info, a.n_questions, a.procs, a.budget_h * 3600)
    print("\n==== PREFLIGHT ====")
    print(f"questions timed: {len(rows)} | sec/question: {rec['sec_per_question']} | model load: {info.get('model_load_s')}s | "
          f"peak VRAM: {info.get('peak_vram_gb')} GB | peak RAM: {info.get('peak_rss_gb')} GB")
    for line in rec["advice"]:
        print("-", line)
    if rec["projected_hours"] is not None:
        print(f"projected full run: {rec['projected_hours']} h on {a.procs} process(es)")
    if a.out:
        json.dump(rec, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
