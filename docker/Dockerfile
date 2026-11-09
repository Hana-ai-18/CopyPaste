# Image chấm offline (không mạng): code + trọng số model đóng sẵn.
#
# !!! BASE IMAGE: dùng đúng image mà hướng dẫn Docker của BTC chỉ định. Máy chấm là 2×RTX 5090 (Blackwell, sm_120):
#     cần CUDA ≥ 12.8 và bản PyTorch/vLLM có kernel sm_120. Image cũ vllm/vllm-openai:v0.6.3 KHÔNG chạy được trên 5090
#     ("no kernel image is available for execution on the device"). Truyền vào lúc build:
#         docker build -f docker/Dockerfile --build-arg BASE_IMAGE=<image theo hướng dẫn BTC> -t datamate .
#     Sau khi build, chạy thử trên GPU Blackwell (hoặc ít nhất kiểm `torch.cuda.get_arch_list()` có 'sm_120').
#
# Chuẩn bị trước khi build (máy có mạng): thư mục models/ chỉ chứa model THẬT SỰ dùng — MỌI model trong image đều tính vào 15B.
#   models/<LLM chính>          — bắt buộc (vd Qwen2.5-Coder-7B-Instruct 7,6B hoặc model ≤ 14B đã đo tốt hơn trên train)
#   models/bge-m3               — tuỳ chọn (0,57B), truy hồi ví dụ đặc tả; không có thì TF-IDF
#   models/PhoGPT-4B-Chat       — tuỳ chọn (3,7B); khuyên KHÔNG đóng gói nếu số đo train không cho thấy lợi
# Kiểm tổng tham số: python scripts/count_params.py models
# Chạy:  docker run --gpus all --network none -v /data:/data:ro -v /result:/result datamate
ARG BASE_IMAGE=REPLACE_WITH_BTC_BASE_IMAGE
FROM ${BASE_IMAGE}
WORKDIR /app
# KHÔNG nâng cấp transformers/torch trong image vLLM (dễ làm hỏng bản vLLM đã dựng sẵn) — chỉ cài gói thuần Python còn thiếu
RUN pip install --no-cache-dir pandas numpy openpyxl python-calamine pyyaml scikit-learn
# chỉ chép code + lexicon (KHÔNG 'COPY . /app': sẽ chép models/ lần nữa vào /app/models -> image gấp đôi, và mọi rác trong thư mục)
COPY *.py /app/
COPY lexicon /app/lexicon
COPY scripts /app/scripts
COPY models /models
ENV MODEL=/models/Qwen2.5-Coder-7B-Instruct \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0 TP=1 SHED=off
# Bật thêm (nếu đã đóng gói): ENV EMBED_MODEL=/models/bge-m3   PHO_MODEL=/models/PhoGPT-4B-Chat
# thiếu kho ví dụ đặc tả -> LLM viết đặc tả không có few-shot (tụt chất lượng âm thầm): chặn build
RUN test -s /app/lexicon/spec_bank.jsonl || (echo "THIẾU lexicon/spec_bank.jsonl — chạy scripts/build_spec_bank.py --train <train> trước khi build" && exit 1)
RUN python /app/datamate_pipeline.py selftest && python /app/scripts/count_params.py /models
ENTRYPOINT ["bash", "/app/scripts/predict.sh", "/data", "/result"]
