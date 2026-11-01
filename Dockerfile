# DataMate - Zalo AI Challenge 2026
# !! Replace BASE_IMAGE with the base image given in the organisers' Docker guideline
#    (the scoring machine has 2x RTX 5090 -> CUDA >= 12.8 and a matching PyTorch/vLLM build).
ARG BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive
WORKDIR /app

COPY requirements.txt requirements-gpu.txt ./
RUN pip install -r requirements.txt && pip install -r requirements-gpu.txt
# vLLM for the scoring machine (backend "vllm"). Pin a version built for CUDA >= 12.8 / Blackwell as in the organisers' guideline.
ARG INSTALL_VLLM=1
RUN if [ "$INSTALL_VLLM" = "1" ]; then pip install vllm; fi

# source code (training + inference), configs, exemplar bank (built from TRAIN only), scripts, README
COPY datamate ./datamate
COPY configs ./configs
COPY scripts ./scripts
COPY notebooks ./notebooks
COPY exemplars ./exemplars
COPY predict.sh README.md ./
# model weights: models/llm is filled by `python scripts/download_model.py --out models/llm` (run it on the host before
# building) or, if missing, downloaded during the build (needs internet at BUILD time only; the container runs offline).
COPY models ./models
ARG DOWNLOAD_MODEL=1
RUN if [ "$DOWNLOAD_MODEL" = "1" ] && [ ! -f models/llm/config.json ]; then python scripts/download_model.py --out models/llm; fi

RUN chmod +x predict.sh
# the organisers mount the test set at /data and read /result/submission.jsonl ; run with --network none
ENTRYPOINT ["./predict.sh"]
