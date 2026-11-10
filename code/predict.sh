#!/usr/bin/env bash
# Reads the test split mounted at /data and writes /result/submission.jsonl (+ /result/logs/).
# Runs fully OFFLINE: no network, no downloads, no external API.
# With >= 2 GPUs (and the vllm/hf backend) one process per GPU is started on a balanced half of the businesses
# (the 8B model fits one GPU), then the shards are merged -> ~2x faster than tensor parallelism for this workload.
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATA_DIR="${DATA_DIR:-/data}"
OUT_DIR="${OUT_DIR:-/result}"
CONFIG="${CONFIG:-configs/rtx5090_vllm.json}"
MODEL_PATH="${MODEL_PATH:-models/llm}"

NGPU="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | grep -c GPU || true)}"
# one process per GPU only when the model fits one GPU (tensor_parallel_size == 1); 14B bf16 uses tensor parallel over both GPUs
TP="$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('vllm',{}).get('tensor_parallel_size',1))" "${CONFIG}" 2>/dev/null || echo 1)"
if [ "${TP}" -gt 1 ]; then NSHARD="${NUM_SHARDS:-1}"; else NSHARD="${NUM_SHARDS:-${NGPU:-1}}"; fi
[ "${NSHARD}" -lt 1 ] && NSHARD=1

mkdir -p "${OUT_DIR}/logs"
if [ "${NSHARD}" -le 1 ]; then
  python -m datamate.run --data_dir "${DATA_DIR}" --out "${OUT_DIR}/submission.jsonl" --logs "${OUT_DIR}/logs" \
    --config "${CONFIG}" --set model_path=\"${MODEL_PATH}\" "$@"
else
  pids=()
  for i in $(seq 0 $((NSHARD - 1))); do
    mkdir -p "${OUT_DIR}/logs/shard${i}"
    CUDA_VISIBLE_DEVICES="${i}" python -m datamate.run --data_dir "${DATA_DIR}" --shard "${i}/${NSHARD}" \
      --out "${OUT_DIR}/shard${i}.jsonl" --logs "${OUT_DIR}/logs/shard${i}" \
      --config "${CONFIG}" --set model_path=\"${MODEL_PATH}\" "$@" > "${OUT_DIR}/logs/shard${i}/stdout.txt" 2>&1 &
    pids+=($!)
  done
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  python scripts/merge_shards.py --questions "${DATA_DIR}/questions.jsonl" --out "${OUT_DIR}/submission.jsonl" \
    $(for i in $(seq 0 $((NSHARD - 1))); do echo "${OUT_DIR}/shard${i}.jsonl"; done)
  [ "$rc" -ne 0 ] && echo "WARNING: a shard failed - its unanswered questions were written as abstain (see logs/shard*/stdout.txt)"
fi
echo "done -> ${OUT_DIR}/submission.jsonl"
