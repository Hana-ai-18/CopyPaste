"""
Sinh notebooks/datamate_kaggle.ipynb (sửa file này rồi chạy lại, đừng sửa tay file .ipynb):

    python notebooks/build_kaggle_notebook.py

Notebook được thiết kế cho "Run All" trên Kaggle (GPU T4 x2, Internet On): mọi bước tự dò, tự dự phòng,
và luôn kết thúc bằng /kaggle/working/submission.jsonl đủ mọi câu public.
Chạy thử cục bộ (không GPU): DM_NB_TEST=1 DM_REPO=<repo git cục bộ> DM_INPUT=<thư mục chứa dữ liệu> DM_WORK=<thư mục ghi>.
"""
import json
import os

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# DataMate (Zalo AI 2026) — chạy trên Kaggle, **Run All** ra file nộp public

**Chuẩn bị 1 lần (bên phải màn hình notebook):**
1. *Settings → Accelerator* = **GPU T4 x2**. *Settings → Internet* = **On** (clone code, cài vLLM, tải model).
2. *Add Input* → dataset `gmnguynhng/copypaste` (dữ liệu train + public_test; thư mục hay file `.zip` đều được).
3. Repo GitHub **private** → *Add-ons → Secrets* thêm `GITHUB_TOKEN` (token chỉ đọc) và bật cho notebook. Repo public thì bỏ qua.
4. Bấm **Run All** (hoặc *Save Version → Save & Run All*).

**Kết quả:** `/kaggle/working/submission.jsonl` (đủ 1 dòng/câu public, đúng thứ tự `questions.jsonl`) + `outputs.zip` (log).

Notebook **không chấm** public (không có và không dùng đáp án public — quy định 6). Chỉ chấm trên TRAIN.
Model: `MODEL` ở ô 0 (mặc định Qwen3.5-9B). Không chạy được trên máy này → tự lùi về `FALLBACK_MODEL` (Qwen2.5-Coder-7B-AWQ),
dòng cuối ô 4 ghi rõ model đang dùng. Mọi bước có dự phòng: cài vLLM lỗi → thử bản khác → backend `hf`; model lỗi giữa chừng → vẫn có đáp án của đường máy tính;
sắp hết 12 giờ → tự dừng và ghi file nộp đủ câu.
""")

md("## 0. Cấu hình (chỉ sửa ô này nếu cần)")
code(r'''
import os, sys, time, json, glob, shutil, zipfile, subprocess, collections, re
T0 = time.time()                       # mốc tính ngân sách thời gian của cả phiên

REPO_URL   = "https://github.com/Hana-ai-18/CopyPaste.git"
BRANCH     = "Gấm"                     # nhánh chứa code pipeline này (nhánh main của repo là code KHÁC). "" = nhánh mặc định
INPUT_DIR  = "/kaggle/input/datasets/gmnguynhng/copypaste"
WORK       = "/kaggle/working"

MODEL      = "Qwen/Qwen3.5-9B"                      # model muốn dùng. Qwen3.5 cần vLLM ≥ 0.17 + 2 GPU (fp16 ~19 GB)
FALLBACK_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"   # MODEL không chạy được trên máy này -> tự lùi về model này (đã chạy ổn)
MODEL_HF   = "Qwen/Qwen2.5-Coder-7B-Instruct"       # chỉ dùng nếu mọi bản vLLM đều lỗi: backend hf (4-bit bitsandbytes, CHẬM)
MAX_LEN    = 28000                     # đủ cho rulebook + thẻ dữ liệu + câu hỏi; prompt dài hơn tự bỏ qua câu đó
N_PATHS    = 2                         # 2 đường độc lập, chỉ trả lời khi đồng thuận
SESSION_HOURS = 11.5                   # Kaggle cắt phiên ở 12 giờ: để dư 30 phút
RUN_SMOKE  = True                      # đo nhanh trên vài câu TRAIN (có điểm) trước khi chạy public
RUN_TRAIN_EVAL = False                 # True: chạy 6 shop TRAIN + fit mô hình bỏ trống (thêm ~1–2 giờ GPU) rồi mới chạy public

# chế độ thử cục bộ (không dùng trên Kaggle)
TEST_MODE = os.environ.get("DM_NB_TEST") == "1"
if TEST_MODE:
    REPO_URL  = os.environ["DM_REPO"]
    INPUT_DIR = os.environ["DM_INPUT"]
    WORK      = os.environ["DM_WORK"]
os.makedirs(WORK, exist_ok=True)
DATA, CACHE = f"{WORK}/data", f"{WORK}/cache"
os.makedirs(DATA, exist_ok=True); os.makedirs(CACHE, exist_ok=True)

def elapsed_h():
    return (time.time() - T0) / 3600

def sh(cmd, check=True, timeout=None, env=None, quiet=False, log=None):
    """Chạy lệnh shell, in log theo thời gian thực, trả (mã thoát, 60 dòng cuối). check=True: lỗi thì dừng notebook.
    log=<đường dẫn>: ghi TOÀN BỘ output vào file (lỗi gốc của vLLM thường nằm xa phía trên 60 dòng cuối)."""
    print(f"$ {cmd if len(cmd) < 400 else cmd[:400] + ' …'}", flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, **(env or {})}, bufsize=1)
    tail = collections.deque(maxlen=60)
    lf = open(log, "w", encoding="utf-8") if log else None
    t = time.time()
    try:
        for line in p.stdout:
            tail.append(line.rstrip("\n"))
            if lf:
                lf.write(line)
            if not quiet:
                print(line, end="", flush=True)
            if timeout and time.time() - t > timeout:
                p.kill(); tail.append(f"[hết {timeout}s → dừng]"); break
        p.wait()
    except KeyboardInterrupt:
        p.kill(); raise
    finally:
        if lf:
            lf.close()
    if quiet and p.returncode:
        print("\n".join(tail))
    if check and p.returncode:
        raise RuntimeError(f"Lệnh lỗi (mã {p.returncode}): {cmd[:200]}")
    return p.returncode, "\n".join(tail)

print("Python", sys.version.split()[0], "| TEST_MODE =", TEST_MODE)
''')

md("## 1. GPU")
code(r'''
rc, out = sh("nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader", check=False)
GPUS = [l for l in out.splitlines() if l.strip() and "," in l] if rc == 0 else []
N_GPU = len(GPUS)
print("Số GPU:", N_GPU)
if not TEST_MODE and N_GPU == 0:
    print("!!! KHÔNG CÓ GPU: vào Settings → Accelerator → GPU T4 x2 rồi Run All lại. "
          "Notebook vẫn chạy tiếp nhưng chỉ có đường máy tính (không LLM).")
''')

md("## 2. Dữ liệu: giải nén (nếu là .zip) và tìm thư mục train / public")
code(r'''
if not os.path.exists(INPUT_DIR):
    print("Không thấy", INPUT_DIR, "→ tìm trong toàn bộ /kaggle/input")
    INPUT_DIR = "/kaggle/input"

def unzip_all(src_root, dst):
    for z in sorted(glob.glob(f"{src_root}/**/*.zip", recursive=True)):
        out = os.path.join(dst, os.path.splitext(os.path.basename(z))[0])
        if not os.path.exists(out):
            print("giải nén", z)
            with zipfile.ZipFile(z) as f:
                f.extractall(out)
unzip_all(INPUT_DIR, DATA)
for _ in range(2):                               # zip lồng nhau
    unzip_all(DATA, DATA)

def find_splits():
    found = []
    for root in (INPUT_DIR, DATA):
        for rb in glob.glob(f"{root}/**/rulebook.md", recursive=True):
            d = os.path.dirname(rb)
            if os.path.exists(f"{d}/questions.jsonl") and os.path.isdir(f"{d}/data") and d not in found:
                found.append(d)
    return found

SPLITS = find_splits()
for d in SPLITS:
    print(d, "| có answers.jsonl:", os.path.exists(f"{d}/answers.jsonl"))
# TRAIN = thư mục CÓ đáp án; PUBLIC = thư mục KHÔNG có đáp án, ưu tiên tên chứa 'public'
TRAIN = next((d for d in SPLITS if os.path.exists(f"{d}/answers.jsonl")), None)
no_ans = [d for d in SPLITS if not os.path.exists(f"{d}/answers.jsonl")]
PUBLIC = next((d for d in no_ans if "public" in d.lower()), no_ans[0] if no_ans else None)
print("TRAIN :", TRAIN)
print("PUBLIC:", PUBLIC)
if not (TRAIN and PUBLIC):
    sh(f"find {INPUT_DIR} -maxdepth 4 | head -60", check=False)
    raise SystemExit("Không tìm thấy train (có answers.jsonl) và public (không có). Mỗi split cần rulebook.md + questions.jsonl + data/.")
for d in (TRAIN, PUBLIC):
    nq = sum(1 for _ in open(f"{d}/questions.jsonl", encoding="utf-8"))
    print(f"{d}: {len(os.listdir(d + '/data'))} shop, {nq} câu")
''')

md("## 3. Lấy code từ GitHub")
code(r'''
CLONE = f"{WORK}/repo"
url = REPO_URL
if not TEST_MODE:
    try:                                             # repo private: token trong Kaggle Secrets
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
        if tok:
            url = REPO_URL.replace("https://", f"https://x-access-token:{tok}@")
            print("Dùng GITHUB_TOKEN từ Kaggle Secrets")
    except Exception:
        print("Không có secret GITHUB_TOKEN → clone như repo public")
def clone(branch):
    shutil.rmtree(CLONE, ignore_errors=True)
    b = f"-b '{branch}'" if branch else ""
    rc, out = sh(f"git clone -q --depth 1 {b} '{url}' '{CLONE}'", check=False, quiet=True)
    return rc, out.replace(url, REPO_URL)            # không in token ra log

def find_code():
    # code có thể nằm ở gốc repo hoặc trong thư mục con (vd khi đẩy cả thư mục datamate-pipeline/)
    return sorted(glob.glob(f"{CLONE}/**/datamate_pipeline.py", recursive=True), key=len)

rc, out = clone(BRANCH)
if rc and BRANCH:
    print(f"Không clone được nhánh '{BRANCH}' → thử nhánh mặc định rồi tự tìm nhánh có code")
    rc, out = clone("")
if rc:
    raise SystemExit("Clone lỗi. Kiểm tra: Internet = On; REPO_URL đúng; repo private thì cần secret GITHUB_TOKEN.\n" + out)
hits = find_code()
if not hits:
    # nhánh đang clone không có pipeline này -> dò mọi nhánh, chỉ tự chọn khi có ĐÚNG MỘT nhánh chứa code
    sh(f"git -C '{CLONE}' fetch -q --depth 1 origin '+refs/heads/*:refs/remotes/origin/*'", check=False, quiet=True)
    _, heads = sh(f"git -C '{CLONE}' for-each-ref --format='%(refname:short)' refs/remotes/origin/", check=False, quiet=True)
    cands = []
    for h in heads.split():
        if h.endswith("/HEAD"):
            continue
        _, files = sh(f"git -C '{CLONE}' ls-tree -r --name-only '{h}'", check=False, quiet=True)
        if any(f.endswith("datamate_pipeline.py") for f in files.split()):
            cands.append(h.split("/", 1)[1])
    print("Nhánh có datamate_pipeline.py:", cands or "không có")
    if len(cands) != 1:
        raise SystemExit("Nhánh đang chọn không có datamate_pipeline.py. Đặt BRANCH ở ô 0 = một trong: " + str(cands))
    print(f"→ tự dùng nhánh '{cands[0]}'")
    rc, out = clone(cands[0])
    hits = find_code()
    if rc or not hits:
        raise SystemExit("Clone nhánh có code lỗi:\n" + out)
sh(f"git -C '{CLONE}' remote set-url origin '{REPO_URL}'", check=False, quiet=True)   # xoá token khỏi .git/config
sh(f"git -C '{CLONE}' branch --show-current", check=False)
CODE = os.path.dirname(hits[0])
os.chdir(CODE)
sh(f"git -C '{CLONE}' log -1 --format='commit %h  %ad  %s' --date=iso", check=False)   # ghi commit đang chạy
need = ["datamate_pipeline.py", "compiler.py", "ledger.py", "rule_delta.py", "retriever.py",
        "lexicon/rulebook_train_fingerprint.json", "scripts/build_spec_bank.py"]
miss = [f for f in need if not os.path.exists(f"{CODE}/{f}")]
print("CODE:", CODE, "| thiếu:", miss or "không")
assert not miss, f"Repo thiếu file: {miss} — đẩy đủ code lên GitHub"
''')

md("## 4. Cài vLLM và chọn model + backend chạy được THẬT trên GPU này\n"
   "Mỗi lần thử: cài bản vLLM hợp với model → nạp model bằng CHÍNH lớp `LLM` của pipeline trong tiến trình riêng → "
   "hỏi `17 + 25` và đòi đúng `42` (bắt cả lỗi fp16 ra chữ rác, model vẫn 'nghĩ' dù đã tắt).\n"
   "Thứ tự: `MODEL` → `FALLBACK_MODEL` → backend `hf` → không LLM. Lần thử thành công cũng đã tải sẵn model.")
code(r'''
def install_plan(model):
    """[(lệnh pip, env, các TP thử)] theo họ model. Qwen3.5 (lai Gated DeltaNet) cần vLLM ≥ 0.17 (chỉ engine V1);
    bản ≥ 0.20 dùng CUDA 13 (driver Kaggle có thể không chạy) nên ghim 0.19.1 / 0.17.1 (CUDA 12.8).
    Qwen2.5: vLLM 0.10.1.1 / 0.8.5 engine V0 (đường chạy quen của T4)."""
    m = model.lower()
    if "qwen3.5" in m or "qwen3_5" in m:
        tps = [2] if N_GPU >= 2 else [1]           # fp16 ~19 GB: 1 T4 (15 GB) không đủ
        return [("'vllm==0.19.1'", {}, tps),
                ("'vllm==0.17.1' 'transformers>=4.57,<5'", {}, tps)]
    tps = [2, 1] if N_GPU >= 2 else [1]
    return [("'vllm==0.10.1.1' 'transformers>=4.55,<4.57'", {"VLLM_USE_V1": "0"}, tps),
            ("'vllm==0.8.5.post1' 'transformers>=4.51,<4.53'", {"VLLM_USE_V1": "0"}, tps)]

PROBE = f"{WORK}/probe_llm.py"
open(PROBE, "w").write(r"""
import sys, time, tempfile
code, backend, model, tp, maxlen = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, code)
import datamate_pipeline as d                  # dùng ĐÚNG lớp LLM mà pipeline dùng (cùng tham số, cùng chat template)
t = time.time()
llm = d.LLM(backend, model, tempfile.mkdtemp(), max_model_len=maxlen, tp=tp)
out = llm.generate([[{"role": "user", "content": "Chỉ trả lời bằng đúng một số, không giải thích: 17 + 25 = ?"}]])[0]
print("OUT", repr(out[:200]), "| vllm_extra =", getattr(llm, "vllm_extra", None))
assert "42" in out, "trả lời sai / rỗng (fp16 hỏng? model vẫn đang nghĩ?)"
print("PROBE_OK", f"{time.time() - t:.0f}s")
""")

def fix_core_libs():
    """Cài vLLM có thể kéo numpy/torch khác -> pandas/sklearn biên dịch theo numpy cũ báo lỗi khi import. Kiểm và sửa."""
    chk = "python -c 'import numpy, pandas, sklearn, openpyxl, yaml; print(numpy.__version__, pandas.__version__, sklearn.__version__)'"
    rc, out = sh(chk, check=False, quiet=True)
    if rc:
        print("pandas/sklearn không hợp numpy mới → cài lại bản hợp")
        sh("pip install -q -U pandas scikit-learn", check=False, quiet=True, timeout=1200)
        rc, out = sh(chk, check=False, quiet=True)
    print("thư viện lõi:", out.strip().splitlines()[-1] if out.strip() else "?", "| OK" if rc == 0 else "| VẪN LỖI")
    return rc == 0

os.makedirs(f"{WORK}/probe_logs", exist_ok=True)

def root_cause(path, n=14):
    """Những dòng lỗi GỐC trong log (bỏ các dòng chuyển tiếp chung chung 'Engine core initialization failed')."""
    pat = re.compile(r"(Error|Exception|error:|Traceback|assert|OutOfResources|out of memory|not supported|Unsupported|CUDA)", re.I)
    skip = re.compile(r"(Engine core initialization failed|See root cause|Traceback \(most recent|File \"|^\s*\^|raise |~~~)")
    seen, out = set(), []
    for l in open(path, encoding="utf-8", errors="replace"):
        l = re.sub(r"^\(\w+ pid=\d+\)\s*", "", l.strip())
        l = re.sub(r"^(ERROR|WARNING|INFO) \d\d-\d\d [\d:]+ \[[^\]]*\]\s*", "", l)
        if l and pat.search(l) and not skip.search(l) and l not in seen:
            seen.add(l); out.append(l[:300])
    return out[:n]

BASE_ENV = {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false"}
RUN_ENV = dict(BASE_ENV)
BACKEND, TP, RUN_MODEL = None, 1, MODEL
TRIED = []
if TEST_MODE or N_GPU == 0:
    BACKEND = "dummy" if TEST_MODE else "none"
else:
    for model in dict.fromkeys([MODEL, FALLBACK_MODEL]):        # bỏ trùng, giữ thứ tự
        for spec, extra_env, tps in install_plan(model):
            rc, _ = sh(f"pip install -q {spec} python-calamine openpyxl pyyaml scikit-learn", check=False, quiet=True,
                       timeout=1800)
            if rc:
                print(f"Cài {spec} lỗi → thử phương án sau"); TRIED.append((model, spec, "cài lỗi")); continue
            fix_core_libs()
            for tp in tps:
                lg = f"{WORK}/probe_logs/{len(TRIED):02d}_{model.split('/')[-1]}_tp{tp}.log"
                rc, out = sh(f"python {PROBE} '{CODE}' vllm {model} {tp} {MAX_LEN}", check=False, quiet=True,
                             timeout=2700, env={**BASE_ENV, **extra_env}, log=lg)
                ok = rc == 0 and "PROBE_OK" in out
                print(f"  {model} | {spec} | TP={tp}: {'CHẠY ĐƯỢC' if ok else 'lỗi'}  (log đầy đủ: {lg})")
                if not ok:
                    print("  LỖI GỐC:\n    " + "\n    ".join(root_cause(lg)))
                TRIED.append((model, spec, tp, "ok" if ok else "lỗi"))
                if ok:
                    BACKEND, TP, RUN_MODEL = "vllm", tp, model
                    RUN_ENV.update(extra_env)
                    break
            if BACKEND:
                break
        if BACKEND:
            break
        print(f"!!! {model} không chạy được trên máy này → thử model sau")
    if BACKEND is None:
        print("!!! vLLM không chạy được → backend hf (chậm hơn nhiều; hết giờ thì câu còn lại = đáp án máy tính)")
        sh("pip install -q -U 'transformers>=4.45' accelerate bitsandbytes python-calamine openpyxl pyyaml scikit-learn",
           check=False, quiet=True, timeout=1800)
        lg = f"{WORK}/probe_logs/hf.log"
        rc, out = sh(f"python {PROBE} '{CODE}' hf {MODEL_HF} 1 {MAX_LEN}", check=False, quiet=True, timeout=2700, log=lg)
        if not (rc == 0 and "PROBE_OK" in out):
            print("  LỖI GỐC (hf):\n    " + "\n    ".join(root_cause(lg)))
        if rc == 0 and "PROBE_OK" in out:
            BACKEND, RUN_MODEL = "hf", MODEL_HF
        else:
            print("!!! Không backend LLM nào chạy được → chỉ dùng đường máy tính (vẫn ra file nộp đủ câu)")
            BACKEND = "none"
for k in ("VLLM_USE_V1",):
    os.environ.pop(k, None)
os.environ.update(RUN_ENV)         # các lệnh python chạy sau (pipeline) kế thừa đúng env của lần thử thành công
sh("pip install -q python-calamine openpyxl pyyaml scikit-learn", check=False, quiet=True)
fix_core_libs()
if RUN_MODEL != MODEL and BACKEND in ("vllm", "hf"):
    print(f"!!! LƯU Ý: đang dùng {RUN_MODEL}, KHÔNG phải MODEL = {MODEL} (MODEL không chạy được trên máy này)")
print(f"BACKEND = {BACKEND} | TP = {TP} | MODEL = {RUN_MODEL} | VLLM_USE_V1 = {os.environ.get('VLLM_USE_V1')}")
''')

md("## 5. Self-test + kho ví dụ đặc tả (dựng từ TRAIN, ~10 phút CPU)")
code(r'''
rc, out = sh(f"python datamate_pipeline.py selftest --split_dir '{TRAIN}'", check=False)
assert "SELFTEST: OK" in out, "Selftest lỗi — xem log ở trên"
SPEC_BANK = f"{CODE}/lexicon/spec_bank.jsonl"
if not (os.path.exists(SPEC_BANK) and os.path.getsize(SPEC_BANK) > 0):
    sh(f"python scripts/build_spec_bank.py --train '{TRAIN}' --cache '{CACHE}/biz' --out '{SPEC_BANK}'")
print("spec_bank:", sum(1 for _ in open(SPEC_BANK, encoding="utf-8")), "ví dụ")
''')

md("## 6. Lệnh chạy chung")
code(r'''
llm_flag = BACKEND                      # vllm | hf | dummy (chỉ thử cục bộ) | none (không LLM: chỉ đáp án chắc của máy tính)
BATCH = 48 if BACKEND == "vllm" else 4
COMMON = (f"--llm {llm_flag} --model {RUN_MODEL} --tp {TP} --max_model_len {MAX_LEN} --n_paths {N_PATHS} "
          f"--batch {BATCH} --cache '{CACHE}' --spec_bank '{SPEC_BANK}' --lm_texts '{TRAIN}/questions.jsonl' "
          f"--exec_timeout 40 --shed off --provisional on")
print(COMMON)
''')

md("## 7. (Tuỳ chọn) Đo nhanh trên TRAIN — có điểm")
code(r'''
if RUN_SMOKE and BACKEND != "none":
    t = time.time()
    lim = 3 if TEST_MODE else 8
    sh(f"python datamate_pipeline.py run --split_dir '{TRAIN}' --out '{WORK}/out_smoke' {COMMON} "
       f"--biz biz_101,biz_119 --limit {lim} --fresh", check=False)
    sh(f"python datamate_pipeline.py score --split_dir '{TRAIN}' --pred '{WORK}/out_smoke/submission.jsonl' --only_pred",
       check=False)
    s = json.load(open(f"{WORK}/out_smoke/run_summary.json", encoding="utf-8"))
    if "loi_nap_model" in s:
        print("!!! Pipeline KHÔNG nạp được model dù bước 4 nạp được — xem lỗi:\n", s["loi_nap_model"][-1500:])
    else:
        print("giây/câu:", s.get("giay_moi_cau"), "| lượt gọi model/câu:", s.get("llm_goi_moi_cau"),
              f"| {(time.time() - t) / 60:.1f} phút")
''')

code(r'''
# (tuỳ chọn) 6 shop TRAIN + fit mô hình bỏ trống theo PA — chỉ dùng TRAIN
ABSTAIN = ""
if RUN_TRAIN_EVAL and BACKEND not in ("none", "dummy"):
    BLIND = "biz_103,biz_108,biz_113,biz_118,biz_124,biz_129"
    budget = max(1800, int((SESSION_HOURS * 0.45 - elapsed_h()) * 3600))
    sh(f"python datamate_pipeline.py run --split_dir '{TRAIN}' --out '{WORK}/out_train' {COMMON} --biz {BLIND} "
       f"--time_budget {budget}", check=False)
    sh(f"python datamate_pipeline.py score --split_dir '{TRAIN}' --pred '{WORK}/out_train/submission.jsonl' --only_pred",
       check=False)
    rc, _ = sh(f"python scripts/fit_abstain.py --train '{TRAIN}' --run '{WORK}/out_train' --out '{WORK}/abstain_model.json'",
               check=False)
    if rc == 0 and os.path.exists(f"{WORK}/abstain_model.json"):
        ABSTAIN = f" --abstain_model '{WORK}/abstain_model.json'"
print("bỏ trống theo mô hình:", ABSTAIN or "không (dùng luật đồng thuận mặc định)")
''')

md("## 8. Chạy PUBLIC TEST → file nộp\n"
   "Ngân sách = thời gian còn lại của phiên. Bị dừng giữa chừng thì `submission.jsonl` vẫn đủ câu "
   "(câu chưa xong = đáp án chắc chắn của đường máy tính hoặc abstain). Chạy lại ô này sẽ **tiếp tục**, không làm lại.")
code(r'''
OUT_PUB = f"{WORK}/out_public"
budget = int((SESSION_HOURS - elapsed_h()) * 3600) - 15 * 60      # để 15 phút cho kiểm file + nén log
print(f"đã dùng {elapsed_h():.2f} giờ | ngân sách public {budget / 3600:.2f} giờ")
lim = " --limit 3" if TEST_MODE else ""
t = time.time()
sh(f"python datamate_pipeline.py run --split_dir '{PUBLIC}' --out '{OUT_PUB}' {COMMON}{ABSTAIN} "
   f"--time_budget {max(budget, 600)}{lim}", check=False)
print(f"public: {(time.time() - t) / 60:.1f} phút")
if os.path.exists(f"{OUT_PUB}/run_summary.json"):
    print(open(f"{OUT_PUB}/run_summary.json", encoding="utf-8").read()[:1500])
''')

md("## 9. Kiểm định dạng + ghi file nộp cuối")
code(r'''
VALID = {"number", "percent", "date", "string", "list", "table", "boolean", "unanswerable", "abstain"}   # = VALID_TYPES của pipeline
qids = [json.loads(l)["id"] for l in open(f"{PUBLIC}/questions.jsonl", encoding="utf-8") if l.strip()]
sub, bad = {}, 0
src = f"{OUT_PUB}/submission.jsonl"
if os.path.exists(src):
    for l in open(src, encoding="utf-8"):
        try:
            r = json.loads(l)
        except Exception:
            bad += 1; continue
        if r.get("type") not in VALID or (r["type"] not in ("abstain", "unanswerable") and "answer" not in r):
            bad += 1; continue
        sub[r["id"]] = r
else:
    print("!!! pipeline không tạo được submission → ghi abstain cho mọi câu (xem log bước 8)")
FINAL = f"{WORK}/submission.jsonl"
with open(FINAL, "w", encoding="utf-8") as f:            # đúng thứ tự câu hỏi, mỗi id 1 dòng, thiếu → abstain
    for q in qids:
        r = sub.get(q, {"id": q, "type": "abstain"})
        f.write(json.dumps({"id": q, **{k: v for k, v in r.items() if k != "id"}}, ensure_ascii=False) + "\n")
rows = [json.loads(l) for l in open(FINAL, encoding="utf-8")]
assert len(rows) == len(qids) and len({r["id"] for r in rows}) == len(qids)
assert not any(k.startswith("_") for r in rows for k in r), "lọt khoá nội bộ"
cnt = collections.Counter(r["type"] for r in rows)
print(f"FILE NỘP: {FINAL} | {len(rows)} câu | dòng hỏng đã bỏ: {bad} | câu thiếu → abstain: {len(qids) - len(sub)}")
print("theo loại:", dict(cnt), f"| trả lời {len(rows) - cnt['abstain']}/{len(rows)}")
shutil.copy(FINAL, f"{WORK}/submission_public.jsonl")
''')

code(r'''
# nén log để tải về (bỏ cache)
dirs = " ".join(d for d in ("out_public", "out_smoke", "out_train", "probe_logs") if os.path.isdir(f"{WORK}/{d}"))
sh(f"cd '{WORK}' && rm -f outputs.zip && zip -qr outputs.zip submission.jsonl {dirs} -x '*/cache/*'", check=False)
print(f"Xong sau {elapsed_h():.2f} giờ. Tải về: {WORK}/submission.jsonl (+ outputs.zip)")
''')

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

if __name__ == "__main__":
    for i, c in enumerate(NB["cells"]):
        c["id"] = f"c{i:02d}"
        c["source"] = c["source"].splitlines(keepends=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datamate_kaggle.ipynb")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(NB, fh, ensure_ascii=False, indent=1)
    print("ghi", out, len(NB["cells"]), "ô")
