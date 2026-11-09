#!/usr/bin/env bash
# Điểm vào cho Docker chấm: đọc /data, ghi /result (submission.jsonl + logs/)
# Biến môi trường có thể đổi: MODEL, TP, N_PATHS, BATCH, MAX_LEN, TIME_BUDGET, SHED,
#   PHO_MODEL (PhoGPT-4B, bật sửa chữ chế độ NORM_LLM=pho|both), EMBED_MODEL (bge-m3), CALIBRATION, ABSTAIN_MODEL
# MỌI model có trong image đều tính vào giới hạn 15B (luật 2) — chỉ đóng gói model thật sự dùng; kiểm bằng scripts/count_params.py
#
# Bảo đảm khi chạy (xem docs/RISKS.md):
#  - submission.jsonl được ghi NGAY sau pha tạm (đường máy tính, không cần GPU) và ghi lại TOÀN BỘ sau mỗi lô:
#    nạp model lỗi / hết 6 giờ / bị ngắt thì file vẫn đủ mọi câu (câu chưa xong = đáp án tạm hoặc abstain);
#  - không có quyết định nào theo tốc độ máy (SHED=off) -> máy BTC chạy lại ra cùng kết quả với máy nhóm;
#  - ngân sách thời gian tính từ lúc tiến trình bắt đầu (gồm nạp model).
set -uo pipefail
DATA_DIR="${1:-/data}"
OUT_DIR="${2:-/result}"
MODEL="${MODEL:-/models/Qwen2.5-Coder-7B-Instruct}"   # trọng số đã đóng sẵn trong image (không có mạng)
TP="${TP:-1}"
N_PATHS="${N_PATHS:-2}"
BATCH="${BATCH:-64}"
MAX_LEN="${MAX_LEN:-28000}"   # prompt dài nhất ~43k ký tự (~17k token) + chỗ cho trả lời và một vòng sửa
TIME_BUDGET="${TIME_BUDGET:-19800}"   # 5,5 giờ TÍNH TỪ LÚC BẮT ĐẦU (gồm nạp model), chừa 30 phút cho giới hạn 6 giờ
SHED="${SHED:-off}"                   # on = tắt dần bước phụ theo tốc độ máy (KHÔNG khuyên: làm kết quả phụ thuộc máy)
EXEC_TIMEOUT="${EXEC_TIMEOUT:-40}"
PHO_MODEL="${PHO_MODEL:-}"                 # vd /models/PhoGPT-4B-Chat ; trống = không dùng PhoGPT
EMBED_MODEL="${EMBED_MODEL:-}"             # vd /models/bge-m3 ; trống = truy hồi bằng TF-IDF
NORM_LLM="${NORM_LLM:-$([ -n "$PHO_MODEL" ] && echo both || echo main)}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CALIBRATION="${CALIBRATION:-$HERE/lexicon/decision_calibration.json}"
ABSTAIN_MODEL="${ABSTAIN_MODEL:-$HERE/lexicon/abstain_model.json}"
mkdir -p "$OUT_DIR/logs"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
# VLLM_BATCH_INVARIANT=1 (kernel bất biến theo batch, beta) — CHỈ bật sau khi đã thử trên 5090 với đúng model; mặc định để nguyên
{
  echo "== môi trường $(date -Is)"
  python -c "import torch,sys;print('torch',torch.__version__,'cuda',torch.version.cuda,'gpu',[torch.cuda.get_device_name(i)+' sm_'+''.join(map(str,torch.cuda.get_device_capability(i))) for i in range(torch.cuda.device_count())],'arch',torch.cuda.get_arch_list())" 2>&1 || true
  python -c "import vllm;print('vllm',vllm.__version__)" 2>&1 || true
  python "$HERE/scripts/count_params.py" /models 2>&1 || true
} > "$OUT_DIR/logs/env.txt" 2>&1
python "$HERE/datamate_pipeline.py" run \
  --split_dir "$DATA_DIR" --out "$OUT_DIR" --cache "$OUT_DIR/cache" \
  --llm vllm --model "$MODEL" --tp "$TP" --max_model_len "$MAX_LEN" \
  ${LM_TEXTS:+--lm_texts "$LM_TEXTS"} \
  --norm_llm "$NORM_LLM" ${PHO_MODEL:+--pho_model "$PHO_MODEL"} ${EMBED_MODEL:+--embed_model "$EMBED_MODEL"} \
  $([ -f "$CALIBRATION" ] && echo --calibration "$CALIBRATION") $([ -f "$ABSTAIN_MODEL" ] && echo --abstain_model "$ABSTAIN_MODEL") \
  --n_paths "$N_PATHS" --batch "$BATCH" --time_budget "$TIME_BUDGET" --shed "$SHED" --exec_timeout "$EXEC_TIMEOUT" \
  2>&1 | tee "$OUT_DIR/logs/run.log"
rc=${PIPESTATUS[0]}
# dù pipeline lỗi giữa chừng, submission.jsonl đã có (pha tạm ghi trước khi nạp model); chỉ báo mã lỗi
[ -s "$OUT_DIR/submission.jsonl" ] || echo "CẢNH BÁO: không có $OUT_DIR/submission.jsonl" | tee -a "$OUT_DIR/logs/run.log"
exit $rc
