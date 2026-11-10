#!/usr/bin/env bash
# Checks the DOCKER PACKAGING end to end on any machine with Docker (no GPU, no model needed):
# builds the image without vLLM / model download, runs it with --network none on the demo split using the mock LLM,
# and validates /result/submission.jsonl. A pass means: files copied, predict.sh works, offline run works, output format valid.
# (It does not test the real model - do that with a real build on the GPU machine.)
set -euo pipefail
cd "$(dirname "$0")/.."
WORK="$(mktemp -d)"
python scripts/make_demo_data.py --out "${WORK}/demo" >/dev/null
docker build -t datamate-selftest --build-arg BASE_IMAGE="${BASE_IMAGE:-python:3.11-slim}" --build-arg INSTALL_VLLM=0 --build-arg DOWNLOAD_MODEL=0 .
mkdir -p "${WORK}/result"
docker run --rm --network none -e CONFIG=configs/default.json -e NUM_GPUS=0 \
  -v "${WORK}/demo:/data:ro" -v "${WORK}/result:/result" datamate-selftest \
  --set backend=\"mock\" --set mock_responses=\"/data/mock_responses.json\"
python scripts/validate_submission.py --sub "${WORK}/result/submission.jsonl" --questions "${WORK}/demo/questions.jsonl"
echo "DOCKER SELFTEST OK ($(wc -l < "${WORK}/result/submission.jsonl") answers)"
