"""Regenerates notebooks/datamate_kaggle.ipynb  (python notebooks/build_notebook.py)."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

MD = lambda s: new_markdown_cell(s.strip("\n"))
CODE = lambda s: new_code_cell(s.strip("\n"))

cells = []

cells.append(MD("""
# DataMate – Zalo AI Challenge 2026 · chạy trên Kaggle / Colab

Notebook này **clone code từ GitHub**, kiểm thử, rồi chạy agent. Chọn chế độ bằng biến `RUN_MODE` ở ô cấu hình:

| `RUN_MODE` | Việc làm | Cần GPU/model |
|---|---|---|
| `smoke` | Chạy unit test + demo end-to-end với LLM giả lập, kiểm tra cả đường ống (≈ 1 phút) | Không |
| `train_eval` | Chạy agent thật trên một lát của **train**, in Penalized Accuracy + gợi ý ngưỡng abstain | Có |
| `build_exemplars` | Chạy agent trên một lát **train** rồi dựng ngân hàng ví dụ few-shot (chỉ từ train) | Có |
| `public_test` | Chạy toàn bộ **public_test**, kiểm tra và tạo `submission.jsonl` để nộp | Có |

**Thiết lập Kaggle (bên phải → Settings):** Accelerator = *GPU T4 x2* · Internet = *On* · *Add Input* = dataset **private** chứa dữ liệu cuộc thi (không đưa dữ liệu lên GitHub/public – thể lệ cấm phát tán).
Repo private: thêm token vào *Add-ons → Secrets* với tên `GITHUB_TOKEN`.
"""))

cells.append(CODE('''
# ======================= CẤU HÌNH – chỉ sửa ô này =======================
GITHUB_REPO   = "https://github.com/Hana-ai-18/CopyPaste.git"   # repo công khai: không cần token (repo private -> thêm Secret GITHUB_TOKEN)
GITHUB_BRANCH = "main"
GITHUB_TOKEN  = ""            # để trống: sẽ lấy từ Kaggle Secret "GITHUB_TOKEN" (nếu có)
REPO_DIR      = ""            # đặt đường dẫn thư mục local để BỎ QUA bước git clone

RUN_MODE      = "smoke,preflight"   # một hoặc NHIỀU chế độ, cách nhau dấu phẩy, chạy lần lượt:
                              #  smoke | preflight | build_exemplars | train_eval | build_exemplars_all | public_test
                              #  gợi ý: phiên 1 "smoke,preflight,build_exemplars,train_eval" · phiên 2 "build_exemplars_all,public_test"
RUN_TESTS     = True          # chạy pytest ở bước kiểm tra
AUTO_TUNE     = True          # dùng kết quả preflight (auto_tune.json): tự giảm context / số lượt bỏ phiếu nếu VRAM sát trần hoặc không kịp giờ
PREFLIGHT_BIZ = ""            # shop dùng để đo (để trống = tự chọn 1 shop của train)
PREFLIGHT_N   = 8             # số câu train thật dùng để đo (RUN_MODE = "preflight")
SESSION_HOURS = 11.0          # giới hạn phiên (Kaggle 12 giờ, chừa dư): agent tự dừng ĐÚNG HẠN, gộp kết quả, rồi chạy lại phiên sau với RESUME
STAGGER_S     = 90            # giãn cách giữa các tiến trình khi nạp model (tránh tràn RAM 29 GB của Kaggle)
NUM_PROCS     = 0             # số tiến trình song song, mỗi tiến trình 1 GPU (0 = tự đếm GPU; T4 x2 -> 2, nhanh gấp đôi)
RESUME        = True          # chạy tiếp từ kết quả đã có (hữu ích khi phiên Kaggle bị ngắt)

MODEL_ID      = "Qwen/Qwen3-8B"   # ""= dùng configs/model.json (Qwen3-14B). Trên Kaggle T4: 8B + t4_kaggle.json (nhanh) hoặc ""/14B + t4_kaggle_14b.json (tốt hơn, chậm hơn)
MODEL_REVISION = ""           # để trống = dùng revision trong configs/model.json (null = bản mới nhất)
MODEL_DIR     = ""            # để trống = /kaggle/temp/models/llm (nếu có) hoặc <WORK>/models/llm; hoặc đường dẫn model có sẵn
CONFIG        = "configs/t4_kaggle.json"           # T4 + Qwen3-8B; dùng configs/t4_kaggle_14b.json nếu MODEL_ID="" (14B); máy RTX 5090: configs/rtx5090_vllm.json

DATA_ROOT     = "/kaggle/input/datasets/gmnguynhng/copypaste"   # thư mục dataset đã up; sai/không tồn tại thì tự dò trong /kaggle/input
HOLDOUT_FOLD = "0/5"          # K/N: giữ lại ~1/5 SỐ SHOP của train làm tập đánh giá (shop test là shop CHƯA THẤY -> phải đo theo shop, không theo câu)
EXEMPLAR_N    = 400           # số câu train (ngoài shop giữ lại) dùng để dựng ví dụ few-shot
EVAL_N        = 150           # số câu của các shop giữ lại dùng để đánh giá / hiệu chỉnh ngưỡng
WORK          = "/kaggle/working"
'''))

cells.append(CODE('''
import os, sys, subprocess, shutil, json, time
T_NB_START = time.time()
from pathlib import Path

def run(cmd, check=True, cwd=None, env=None):
    """Run a command and stream its output into the notebook."""
    shell = isinstance(cmd, str)
    print("$", cmd if shell else " ".join(map(str, cmd)), flush=True)
    e = dict(os.environ, PYTHONHASHSEED="0", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    if env:
        e.update(env)
    if cwd is None and "REPO" in globals():
        cwd = str(REPO)                      # always run from the repo, whatever the kernel's current directory is
    p = subprocess.Popen(cmd, shell=shell, cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="")
    rc = p.wait()
    if check and rc != 0:
        raise RuntimeError(f"command failed (exit code {rc})")
    return rc

print("python", sys.version.split()[0])
run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo 'no GPU visible'", check=False)
WORK = Path(WORK)
WORK.mkdir(parents=True, exist_ok=True)
MODES = [m.strip() for m in RUN_MODE.split(",") if m.strip()]
VALID = ("smoke", "preflight", "build_exemplars", "train_eval", "build_exemplars_all", "public_test")
assert MODES and all(m in VALID for m in MODES), f"RUN_MODE is invalid: {RUN_MODE!r}; valid: {VALID}"
NEEDS_GPU = any(m != "smoke" for m in MODES)
'''))

cells.append(MD("## 1 · Lấy code từ GitHub"))
cells.append(CODE('''
if REPO_DIR:
    REPO = Path(REPO_DIR).resolve()
else:
    token = GITHUB_TOKEN
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("GITHUB_TOKEN")
        except Exception:
            token = ""
    url = GITHUB_REPO
    if token and url.startswith("https://"):
        url = url.replace("https://", f"https://{token}@", 1)
    REPO = WORK / "repo"
    if REPO.exists():
        shutil.rmtree(REPO)
    r = subprocess.run(["git", "clone", "--depth", "1", "--branch", GITHUB_BRANCH, url, str(REPO)], capture_output=True, text=True)
    if r.returncode != 0:
        err = r.stderr.replace(token, "***") if token else r.stderr
        raise RuntimeError("git clone failed (check GITHUB_REPO / branch / token):\\n" + err)
    subprocess.run(["git", "-C", str(REPO), "remote", "set-url", "origin", GITHUB_REPO])   # do not keep the token in .git/config
os.chdir(REPO)
sys.path.insert(0, str(REPO))
_missing = [f for f in ("scripts/download_model.py", "scripts/merge_shards.py", "datamate/run.py", "configs/t4_kaggle.json") if not (REPO / f).exists()]
if _missing:
    raise FileNotFoundError(f"repo {REPO} is missing {_missing}: push the whole project (scripts/, configs/, datamate/ ...) to GitHub, then re-run this cell")
# the GPU run only works with the chunked-prefill fix: refuse to start on an outdated repo (a GitHub push is needed after every update)
_llm_src = (REPO / "datamate" / "llm.py").read_text(encoding="utf-8")
if "prefill_chunk" not in _llm_src or "prefix_cache" not in _llm_src:
    raise RuntimeError("The GitHub repo is OUTDATED (datamate/llm.py has no chunked prefill -> every question would hit CUDA out-of-memory on a T4). "
                       "Unzip CopyPaste_update3.zip over the repo folder, then: git add -A && git commit -m update && git push, and run this notebook again.")
from datamate.session import import_previous_outputs, remaining_budget_s
for line in import_previous_outputs("/kaggle/input", WORK):       # continue a previous session (its output attached via Add Data -> Notebook Output)
    print("restored previous run:", line)
# exemplar banks built in an earlier session and uploaded as a (private) Kaggle dataset are picked up automatically
for f in list(Path("/kaggle/input").rglob("exemplars*.jsonl")) if Path("/kaggle/input").exists() else []:
    dst = REPO / "exemplars" / f.name
    if not dst.exists():
        dst.parent.mkdir(exist_ok=True)
        shutil.copy(f, dst)
        print("imported", f.name, "from", f.parent)
sys.path.insert(0, str(REPO))
print("repo:", REPO)
run("git log -1 --oneline || true", check=False)
'''))

cells.append(MD("## 2 · Cài thư viện và chạy kiểm thử"))
cells.append(CODE('''
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
if NEEDS_GPU:
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-gpu.txt"])
if RUN_TESTS:
    run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests/test_textutil_scoring.py", "tests/test_helpers_loader.py",
         "tests/test_sandbox_agent.py", "tests/test_pipeline.py", "tests/test_misc.py"])
'''))

cells.append(MD("""
## 3 · Smoke test (không cần GPU/model)
Sinh dữ liệu demo (3 doanh nghiệp: csv / xlsx / sqlite), chạy **đúng CLI thật** với LLM giả lập có chủ ý sai/lỗi để kiểm tra sửa lỗi, bỏ phiếu, abstain, rồi chấm điểm.
Kỳ vọng: **PA ≈ 0.92** (12/13 đúng, 1 abstain có chủ đích).
"""))
cells.append(CODE('''
demo = WORK / "demo_split"
run([sys.executable, "scripts/make_demo_data.py", "--out", str(demo)])
smoke = WORK / "smoke"
shutil.rmtree(smoke, ignore_errors=True)
run([sys.executable, "-m", "datamate.run", "--data_dir", str(demo), "--out", str(smoke / "submission.jsonl"), "--logs", str(smoke / "logs"),
     "--config", "configs/default.json", "--set", 'backend="mock"', "--set", f'mock_responses="{demo / "mock_responses.json"}"'])
run([sys.executable, "scripts/validate_submission.py", "--sub", str(smoke / "submission.jsonl"), "--questions", str(demo / "questions.jsonl")])
from datamate.evaluate import report
m = report(smoke / "submission.jsonl", demo / "answers.jsonl")
assert m["PA"] > 0.9 and m["wrong"] == 0, "SMOKE TEST FAILED - do not continue"
print("\\nSMOKE TEST OK")
'''))

cells.append(MD("## 4 · Model và dữ liệu thật (bỏ qua khi chỉ chạy `smoke`)"))
cells.append(CODE('''
if NEEDS_GPU:
    os.chdir(REPO)                           # relative paths (scripts/, configs/) must resolve even after a kernel restart / re-run
    if MODEL_DIR:
        mdir = Path(MODEL_DIR)
    else:   # pick the disk with the most free space; /kaggle/working is only ~20 GB (and would be saved as notebook output)
        cands = [Path(p) for p in ("/kaggle/temp", "/tmp", str(WORK)) if Path(p).is_dir()]
        best = max(cands, key=lambda p: shutil.disk_usage(p).free)
        mdir = best / "models" / "llm"
        print("free disk (GB):", {str(p): round(shutil.disk_usage(p).free / 1e9) for p in cands}, "-> model in", mdir)
    if "awq" in MODEL_ID.lower():          # vLLM + AWQ 4-bit: batches the 3-5 variants of a question and caches the shared prompt (~5-10x faster than transformers on a T4)
        CONFIG = "configs/t4_vllm_14b.json" if "14b" in MODEL_ID.lower() else "configs/t4_vllm.json"
        print("MODEL_ID is an AWQ model -> CONFIG =", CONFIG)
    if "vllm" in CONFIG:
        run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
        _v = subprocess.run([sys.executable, "-c", "import vllm; print(vllm.__version__)"], capture_output=True, text=True)
        if _v.returncode != 0:
            raise RuntimeError("vLLM cannot be imported here:\\n" + _v.stderr[-1500:] + "\\nUse MODEL_ID='Qwen/Qwen3-8B' with CONFIG='configs/t4_kaggle.json' (transformers backend, slower).")
        print("vllm", _v.stdout.strip())
    if "14b" in MODEL_ID.lower() and CONFIG.endswith("t4_kaggle.json"):
        CONFIG = "configs/t4_kaggle_14b.json"
        print("MODEL_ID is 14B -> CONFIG switched to", CONFIG)
        STAGGER_S = max(STAGGER_S, 240)   # 30 GB of weights per process: let the first one finish loading before the second starts (Kaggle RAM ~29 GB)
    if "8b" in MODEL_ID.lower() and CONFIG.endswith("t4_kaggle_14b.json"):
        CONFIG = "configs/t4_kaggle.json"
        print("MODEL_ID is 8B -> CONFIG switched to", CONFIG)
    MODEL_DIR = str(mdir)
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            hf_token = ""
    cmd = [sys.executable, "scripts/download_model.py", "--out", str(mdir)]
    if MODEL_ID:
        cmd += ["--model_id", MODEL_ID]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    # needs "Internet: On" in the Kaggle notebook settings (skipped if the folder already contains config.json)
    run(cmd, env={"HF_TOKEN": hf_token} if hf_token else None)   # also checks the 15B parameter limit (exit code != 0 -> stop)
    gb = sum(f.stat().st_size for f in mdir.rglob("*") if f.is_file()) / 1e9
    print(f"model at {mdir}  ({gb:.1f} GB)")

    def find_splits(root):
        found = {}
        for q in Path(root).rglob("questions.jsonl"):
            d = q.parent
            if not (d / "data").is_dir():
                continue
            low = str(d).lower()
            kind = "train" if (d / "answers.jsonl").exists() else "private_test" if "private" in low else "public_test" if ("public" in low or "test" in low) else "unknown"
            found.setdefault(kind, d)
        return found

    import zipfile   # the organisers ship train.zip / public_test.zip: unzip them (also inside an outer zip) once into WORK/data
    def unzip_all(root, dest, depth=0):
        for z in Path(root).rglob("*.zip"):
            target = dest / z.stem
            if target.exists() or depth > 2 or "/kaggle/working/models" in str(z):
                continue
            try:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(target)
                print("unzipped", z, "->", target)
                if depth < 2:
                    unzip_all(target, dest, depth + 1)
            except Exception as e:
                print("cannot unzip", z, e)
    if DATA_ROOT and not Path(DATA_ROOT).exists():
        print(f"DATA_ROOT {DATA_ROOT} không tồn tại -> tự dò trong /kaggle/input")
        DATA_ROOT = ""
    for root in ([DATA_ROOT] if DATA_ROOT else ["/kaggle/input", "/content"]):
        if Path(root).exists():
            unzip_all(root, WORK / "data")

    SPLITS = {}
    for root in ([DATA_ROOT, str(WORK / "data")] if DATA_ROOT else ["/kaggle/input", "/content", str(WORK / "data"), str(WORK), "."]):
        if Path(root).exists():
            for k, v in find_splits(root).items():
                SPLITS.setdefault(k, v)
    print("splits found:", {k: str(v) for k, v in SPLITS.items()})
'''))

cells.append(MD("""
## 5 · Chạy lần lượt các chế độ trong `RUN_MODE`

> **Thời gian:** trên T4 mỗi câu hỏi có thể mất vài chục giây đến vài phút (sinh 3–5 lời giải). Một phiên Kaggle tối đa ~12 giờ, nên với `public_test` hãy dùng `RESUME = True` và chạy lại ô này nếu phiên bị ngắt, hoặc giảm số lời giải bằng `--set agent.stages=[1]` khi chỉ muốn thử nhanh.
> Để so sánh các cấu hình với nhau hãy dùng `train_eval` (có đáp án), đừng đốt lượt nộp (5/ngày).
"""))
cells.append(CODE('''
os.chdir(REPO)
def need(kind):
    if kind not in SPLITS:
        raise FileNotFoundError(f"split '{kind}' not found. Set DATA_ROOT to the folder that contains {kind}/questions.jsonl and {kind}/data/")
    return SPLITS[kind]

def agent_cmd(split, out, extra=()):
    cmd = [sys.executable, "-m", "datamate.run", "--data_dir", str(split), "--out", str(out / "submission.jsonl"), "--logs", str(out / "logs"),
           "--config", CONFIG, "--set", f'model_path="{MODEL_DIR}"', *extra]
    if RESUME:
        cmd.append("--resume")
    return cmd

def n_gpus():
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        return sum(1 for l in r.stdout.splitlines() if l.startswith("GPU"))
    except Exception:
        return 0

def run_agent(split, out, extra=(), questions_file=None):
    """One process per GPU (the model fits one T4 in 4-bit): shard i/N on GPU i, then merge. N=1 -> the plain single run."""
    import threading
    n = NUM_PROCS or max(1, n_gpus())
    extra = list(extra) + ["--set", f"time_budget_s={remaining_budget_s(SESSION_HOURS, T_NB_START)}"]   # stop in time, never get killed mid-write
    tune = WORK / "auto_tune.json"
    if AUTO_TUNE and tune.exists():
        for kv in json.loads(tune.read_text()).get("set", []):
            extra += ["--set", kv]
            print("auto-tune:", kv)
    if n <= 1:
        return run(agent_cmd(split, out, extra))
    extra = [x for x in extra if x != "--eval"]          # scoring happens once, after the merge
    procs, files = [], []
    for i in range(n):
        o = out / f"shard{i}"
        o.mkdir(parents=True, exist_ok=True)
        cmd = agent_cmd(split, o, [*extra, "--shard", f"{i}/{n}"])
        e = dict(os.environ, PYTHONHASHSEED="0", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES=str(i))
        print(f"[shard {i}/{n}] on GPU {i}:", " ".join(map(str, cmd[3:8])), "...", flush=True)
        if i and STAGGER_S:
            time.sleep(STAGGER_S)
        procs.append(subprocess.Popen(cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1))
        files.append(o / "submission.jsonl")
    def pump(i, p):
        for line in p.stdout:
            if i == 0 or "Error" in line or "Traceback" in line or "STOP" in line:
                print(f"[{i}] {line}", end="")
    ts = [threading.Thread(target=pump, args=(i, p)) for i, p in enumerate(procs)]
    [t.start() for t in ts]
    rcs = [p.wait() for p in procs]
    [t.join() for t in ts]
    if any(rcs):
        print("WARNING: a shard exited with", rcs, "- its unanswered questions become abstain; re-run this cell with RESUME=True to continue")
    merge = [sys.executable, "scripts/merge_shards.py", "--out", str(out / "submission.jsonl")]
    if questions_file:
        merge += ["--questions", str(questions_file)]
    run([*merge, *map(str, files)])
    (out / "logs").mkdir(parents=True, exist_ok=True)
    with open(out / "logs" / "log.jsonl", "w", encoding="utf-8") as fo:      # one combined log for evaluate / build_exemplars
        for i in range(n):
            lf = out / f"shard{i}" / "logs" / "log.jsonl"
            if lf.exists():
                fo.write(lf.read_text(encoding="utf-8"))

for MODE in MODES:
    print('\\n' + '=' * 20, 'MODE:', MODE, '=' * 20)
    if MODE == "preflight":
        # measure on the REAL GPU with a few real questions: model load time, sec/question, peak VRAM/RAM -> settings for the long run
        split, out = need("train"), WORK / "preflight"
        # ONE shop (loading a shop costs 0.5-2.5 min): the questions share the prompt prefix exactly as in the real run, so the measured
        # sec/question includes prefix reuse. A shop that belongs to the hold-out fold is avoided (no exemplars are used here anyway).
        PREFLIGHT_BIZ = PREFLIGHT_BIZ or sorted({json.loads(l)["business_id"] for l in open(split / "questions.jsonl", encoding="utf-8") if l.strip()})[1]
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        cmd = [sys.executable, "-m", "datamate.run", "--data_dir", str(split), "--out", str(out / "submission.jsonl"), "--logs", str(out / "logs"),
               "--config", CONFIG, "--set", f'model_path="{MODEL_DIR}"', "--set", 'agent.exemplars_path="none"',
               "--business", PREFLIGHT_BIZ, "--limit", str(PREFLIGHT_N), "--eval"]
        run(cmd, env={"CUDA_VISIBLE_DEVICES": "0"}, check=False)
        target = SPLITS["public_test"] / "questions.jsonl" if "public_test" in SPLITS else None
        n_target = sum(1 for _ in open(target, encoding="utf-8")) if target else 1000
        run([sys.executable, "-m", "datamate.preflight", "--logs", str(out / "logs"), "--n_questions", str(n_target), "--procs", str(NUM_PROCS or max(1, n_gpus())),
             "--budget_h", str(max(1.0, SESSION_HOURS - 0.7)), "--out", str(WORK / "auto_tune.json")])
        print("Saved", WORK / "auto_tune.json", "- the next runs apply it automatically (AUTO_TUNE = True).")

    elif MODE == "train_eval":
        split, out = need("train"), WORK / "train_eval"
        out.mkdir(parents=True, exist_ok=True)
        # evaluate on SHOPS the exemplar bank has never seen (exemplars_safe.jsonl is built without them): an honest estimate for the unseen test shops
        safe_bank = Path("exemplars/exemplars_safe.jsonl")
        extra = ["--biz_fold", HOLDOUT_FOLD, "--limit", str(EVAL_N), "--eval"]
        if safe_bank.exists():
            extra += ["--set", f'agent.exemplars_path="{safe_bank}"']
        else:
            extra += ["--set", 'agent.exemplars_path="none"']
            print("no exemplars_safe.jsonl yet: evaluating zero-shot (run the 'build_exemplars' mode first for the real estimate)")
        run_agent(split, out, extra)
        run([sys.executable, "-m", "datamate.evaluate", "--pred", str(out / "submission.jsonl"), "--gold", str(split / "answers.jsonl"),
             "--calibrate", str(out / "logs" / "log.jsonl")])

    elif MODE in ("build_exemplars", "build_exemplars_all"):
        split, out = need("train"), WORK / "exemplar_run"
        out.mkdir(parents=True, exist_ok=True)
        scope_args = [] if MODE == "build_exemplars_all" else ["--exclude_fold", HOLDOUT_FOLD]
        target = "exemplars/exemplars.jsonl" if MODE == "build_exemplars_all" else "exemplars/exemplars_safe.jsonl"
        run_agent(split, out, [*scope_args, "--limit", str(EXEMPLAR_N), "--set", 'agent.exemplars_path="none"'])
        run([sys.executable, "scripts/build_exemplars.py", "--log", str(out / "logs" / "log.jsonl"), "--gold", str(split / "answers.jsonl"),
             "--out", target])
        print("built", target)
        print("exemplars/exemplars.jsonl is built from TRAIN only. Keep it out of git (it contains train questions); it is copied into the Docker image.")

    elif MODE == "public_test":
        split, out = need("public_test"), WORK / "public_test_out"
        out.mkdir(parents=True, exist_ok=True)
        run_agent(split, out, questions_file=split / "questions.jsonl")
        run([sys.executable, "scripts/validate_submission.py", "--sub", str(out / "submission.jsonl"), "--questions", str(split / "questions.jsonl")])
        shutil.copy(out / "submission.jsonl", WORK / "submission.jsonl")
        print("\\n-> /kaggle/working/submission.jsonl is ready: download it and upload it in the 'Submission' section of the portal.")
    elif MODE == "smoke":
        pass   # smoke already ran above
'''))

cells.append(MD("""
## 6 · Xem nhanh kết quả
Mở `logs/log.jsonl` để xem mã do LLM sinh ra, kết quả thực thi, độ đồng thuận (`confidence`) và quyết định của từng câu.
"""))
cells.append(CODE('''
import pandas as pd
cand = [WORK / "public_test_out", WORK / "train_eval", WORK / "exemplar_run", WORK / "smoke"]
out = next((p for p in cand if (p / "logs" / "log.jsonl").exists()), None)
if out is None:
    print("no run logs yet")
else:
    rows = [json.loads(l) for l in open(out / "logs" / "log.jsonl", encoding="utf-8") if l.strip()]
    df = pd.DataFrame([{"id": r["id"], "decision": r.get("decision"), "confidence": r.get("confidence"), "type": r["submitted"]["type"],
                        "elapsed_s": r.get("elapsed_s"), "question": r["question"][:60]} for r in rows])
    print(out)
    print(df["decision"].value_counts().to_string())
    print("mean seconds / question:", round(df["elapsed_s"].mean(), 1))
    display(df.head(15))
'''))

for i, c in enumerate(cells):
    c["id"] = f"cell-{i:02d}"  # fixed ids -> the build is deterministic (clean git diffs, stable test)

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
nbformat.validate(nb)
out = Path(__file__).with_name("datamate_kaggle.ipynb")
nbformat.write(nb, out)
print("wrote", out)
