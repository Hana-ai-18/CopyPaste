# DataMate – Zalo AI Challenge 2026

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng (CSV / Excel / SQLite) của hộ kinh doanh nhỏ.
**LLM không tự tính số.** LLM viết code pandas/SQL → code chạy trong sandbox trên dữ liệu thật → nhiều lời giải được bỏ phiếu → chỉ trả lời khi đủ chắc, nếu không thì `abstain` (vì đáp án sai bị −1, abstain là 0).

> **Chạy trên Kaggle (GPU): xem [HUONG_DAN_KAGGLE.md](HUONG_DAN_KAGGLE.md).**

## 1. Kiến trúc

```
question + as_of_date ─┐
rulebook.md, calendar.csv, README.md, hồ sơ dữ liệu (profile) ─┴─► prompt ─► LLM (cục bộ, greedy)
        ▲                                                              │  N lời giải (3, nếu bất đồng thì 5)
        │ ví dụ few-shot (chỉ từ train)                                ▼
   ExemplarBank                              Sandbox (worker riêng, fork + timeout, dữ liệu chỉ đọc)
                                                       │ lỗi → LLM sửa code (tối đa 2 vòng)
                                                       ▼
                         chuẩn hóa đáp án theo type ─► bỏ phiếu ─► độ đồng thuận (confidence)
                                      confidence ≥ ngưỡng ? trả lời / unanswerable : abstain
```

| Thành phần | File | Việc làm |
|---|---|---|
| Nạp dữ liệu | `datamate/data_loader.py` | csv (tự dò dấu phân cách/encoding), xlsx (tự tìm dòng tiêu đề thật, giữ kiểu ngày), sqlite (chỉ đọc) |
| Hồ sơ dữ liệu | `datamate/profile.py` | cột, kiểu, giá trị phân loại (status…), khoảng ngày, cột ngày dạng text, gợi ý khóa nối |
| Helper nghiệp vụ | `datamate/helpers.py` | tuần T2–CN, tháng-to-date, cùng kỳ, quý/năm, `parse_dates` an toàn, `rank_top` (đồng hạng → mã tăng dần), làm tròn half-up, Tết 23 Chạp–mùng 10 từ `calendar.csv` |
| Sandbox | `datamate/sandbox.py` | worker `spawn` khởi động **trước** khi nạp GPU, mỗi lần chạy `fork` + timeout; chặn `import os/subprocess/socket`, chặn `open` |
| Chuẩn hóa đáp án | `datamate/answer_format.py` | numpy/pandas → JSON, làm tròn đúng quy tắc, khóa chuẩn để bỏ phiếu |
| Agent | `datamate/agent.py`, `prompts.py` | 5 biến thể prompt, sửa lỗi, bỏ phiếu theo giai đoạn (dừng sớm nếu 3/3 đồng ý), ngưỡng abstain/unanswerable |
| LLM | `datamate/llm.py` | `hf` (transformers 4-bit, chạy được trên T4), `vllm` (máy chấm), `mock` (chỉ để test) |
| Chạy + log | `datamate/run.py` | ghi từng dòng ngay, `--resume`, ngân sách thời gian, mọi lỗi → `abstain`, log từng câu |
| Chấm cục bộ | `datamate/scoring.py`, `evaluate.py` | bản cài lại đúng thể lệ (PA, từng type) + `--calibrate` chọn ngưỡng |

## 2. Đối chiếu thể lệ (DataMate)

| Yêu cầu BTC | Cách đáp ứng |
|---|---|
| Offline, không API ngoài | Model nạp từ thư mục trong image; `predict.sh` đặt `HF_HUB_OFFLINE=1`; chạy Docker với `--network none` |
| Tổng model ≤ 15B, phát hành trước 01/10/2026 | Dùng **1** model. **Bạn phải tự kiểm tra** số tham số, ngày phát hành, giấy phép và ghi vào mục 7 |
| Không người trong vòng lặp, không hard-code theo `id`/`business_id` | Không có logic nào dựa vào id. Ví dụ few-shot chọn theo độ giống của **nội dung câu hỏi** |
| Không dùng test để train/gán nhãn | `build_exemplars.py` **từ chối** đường dẫn có `public_test`/`private_test`. Notebook dựng ví dụ chỉ từ `train` |
| Đọc `rulebook.md` của đúng split | Rulebook của split đang chạy được đưa vào mọi prompt và vào biến `rulebook` trong sandbox |
| `/data` chỉ đọc | Chỉ đọc dữ liệu; sqlite mở `mode=ro`; file tạm không ghi vào `/data` |
| ≤ 6 giờ cho private test (kể cả nạp model) | `time_budget_s` (mặc định 5h15m) tính từ lúc tiến trình bắt đầu; hết ngân sách → các câu còn lại `abstain`; tự chuyển "fast mode" nếu bình quân thời gian/câu quá thấp |
| `predict.sh` đọc `/data`, ghi `/result/submission.jsonl` | Có sẵn. Ghi **từng dòng** ngay khi có đáp án, rồi ghi lại đủ thứ tự cuối cùng (thiếu câu nào → `abstain`) |
| Log từng câu (id, đáp án, thời gian, code, kết quả chạy) | `/result/logs/log.jsonl` + `run_info.json` (cấu hình, seed, phiên bản) |
| Tái lập, lệch PA ≤ 0.03 | Seed cố định, giải mã greedy, `PYTHONHASHSEED=0`, biến thể prompt thay cho lấy mẫu ngẫu nhiên |
| Khai báo model trong README | Mục 7 |

## 3. Chạy nhanh (không cần GPU)

```bash
pip install -r requirements-dev.txt
python -m pytest -q                                   # ~1 phút, gồm cả chạy notebook
python scripts/make_demo_data.py --out demo_split
python -m datamate.run --data_dir demo_split --out out/sub.jsonl --logs out/logs \
   --set backend=\"mock\" --set mock_responses=\"demo_split/mock_responses.json\" --eval
```
Kết quả mong đợi: `Penalized Accuracy = 0.9231` (12 đúng, 1 abstain **có chủ đích**). Con số này kiểm tra *đường ống*, **không** phản ánh chất lượng thật của bất kỳ LLM nào.

## 4. Quy trình làm việc đề xuất

1. **Dev trên train** (có `answers.jsonl`), không đốt lượt nộp:
   ```bash
   python -m datamate.run --data_dir train --out out/train/sub.jsonl --logs out/train/logs \
       --config configs/t4_kaggle.json --set model_path=\"models/llm\" --offset 300 --limit 60 --eval
   python -m datamate.evaluate --pred out/train/sub.jsonl --gold train/answers.jsonl --calibrate out/train/logs/log.jsonl
   ```
   Xem `out/train/logs/log.jsonl` để biết loại câu nào sai nhiều (cột sai tên, quy tắc rulebook chưa đúng, định dạng đáp án…) rồi sửa `prompts.py`/`helpers.py`.
2. **Dựng ví dụ few-shot từ train** (lát đầu, tách khỏi lát đánh giá để không rò rỉ):
   ```bash
   python -m datamate.run --data_dir train --out out/ex/sub.jsonl --logs out/ex/logs --config configs/t4_kaggle.json --set model_path=\"models/llm\" --limit 300
   python scripts/build_exemplars.py --log out/ex/logs/log.jsonl --gold train/answers.jsonl --out exemplars/exemplars.jsonl
   ```
3. **Hiệu chỉnh ngưỡng** `agent.threshold_answer` / `threshold_unanswerable` bằng `--calibrate` (mặc định 0.6 / 0.8). Mẫu nhỏ dễ overfit: so với mặc định rồi chọn giá trị hợp lý.
4. **Public test:** chạy, rồi `python scripts/validate_submission.py --sub ... --questions public_test/questions.jsonl` **trước khi nộp** (5 lượt/ngày).
5. **Đóng gói Docker** (hạn 16/11 23:59), kiểm tra offline:
   ```bash
   python scripts/download_model.py --out models/llm --pin     # tải model trong configs/model.json, ghim commit, kiểm tra ≤15B
   docker build --build-arg BASE_IMAGE=<base image theo hướng dẫn BTC> -t datamate .
   docker run --rm --gpus all --network none -v "$PWD/public_test:/data:ro" -v "$PWD/result:/result" datamate
   python scripts/validate_submission.py --sub result/submission.jsonl --questions public_test/questions.jsonl
   ```
   Đo thời gian trên GPU gần với 2× RTX 5090; so điểm với bản chạy ở Kaggle (lệch ≤ 0.03 mới an toàn).

## 5. Đẩy lên GitHub (để Kaggle clone)

```bash
cd zaic2026-datamate
git init && git add -A && git status        # kiểm tra: KHÔNG có train/, public_test/, models/, exemplars/*.jsonl
git commit -m "DataMate agent"
git branch -M main
git remote add origin https://github.com/<ban>/<repo>.git     # nên để repo PRIVATE
git push -u origin main
```
`.gitignore` đã loại dữ liệu cuộc thi (thể lệ cấm phát tán), model, kết quả và ngân hàng ví dụ (chứa câu hỏi train). Mở `notebooks/datamate_kaggle.ipynb` trên Kaggle, sửa ô cấu hình (`GITHUB_REPO`, `RUN_MODE`…) rồi Run All.

## 6. Giới hạn đã biết (đọc kỹ)

- **Chưa chạy với dữ liệu và model thật.** Toàn bộ test dùng dữ liệu demo và LLM giả lập. Backend `hf`/`vllm` chỉ được kiểm tra bằng stub (không có GPU/Hugging Face trong môi trường phát triển) → lần đầu chạy trên Kaggle hãy dùng `RUN_MODE = train_eval` với vài câu trước.
- **Định dạng `calendar.csv` chưa biết.** Code tự dò cột (ngày dương, tháng/ngày âm, nhuận). Nếu dò thất bại, `tet_window()` báo lỗi rõ ràng nhưng LLM vẫn dùng được `calendar_df` thô. Hãy mở `calendar.csv` thật và kiểm tra bằng `LunarCalendar` (xem `tests/test_helpers_loader.py`).
- **Tốc độ trên T4 chậm** (4-bit, nhiều lời giải/câu). Dùng `--resume`, `agent.stages=[1]` khi chỉ cần thử, hoặc `batch_size` lớn hơn (kết quả có thể lệch số học rất nhỏ so với `batch_size=1`).
- **Biên dịch đáp án dạng `string`:** hệ thống chấp nhận cả tên và mã hàng, nhưng bỏ phiếu coi hai dạng là khác nhau; prompt yêu cầu luôn trả **tên** trừ khi câu hỏi đòi mã.
- Sandbox chống sai sót chứ không phải ranh giới bảo mật; ranh giới thật là container offline.

## 7. Model sử dụng

| Model | Số tham số | Ngày phát hành | Giấy phép | Lượng tử hóa | Dùng để |
|---|---|---|---|---|---|
| `Qwen/Qwen3-14B` (khai báo trong `configs/model.json`) | ~14,8B – **sát giới hạn 15B**, `model_check` đếm số thật | ~04/2025 (**tự kiểm tra**) | Apache-2.0 (**tự kiểm tra**) | bf16 chia 2 GPU (tensor parallel) trên 2×RTX 5090; 4-bit trên T4 (chỉ để thử) | sinh code pandas/SQL |

- **Trọng số KHÔNG nằm trong git** (GitHub giới hạn 100 MB/file). Repo chỉ ghim `model_id` + `revision` trong `configs/model.json`; notebook tự chạy `scripts/download_model.py` (cần bật **Internet: On** trên Kaggle; model lưu ở `/kaggle/temp` để không chiếm 20 GB của `/kaggle/working`).
- Chạy `python scripts/download_model.py --pin` một lần rồi commit `configs/model.json` để cố định commit (tái lập kết quả).
- `datamate.run` tự đếm tham số từ header safetensors, ghi vào `run_info.json` và **dừng** nếu vượt `max_model_params_b` (15).
- Qwen3 cần `transformers >= 4.51` (đã đặt trong `requirements-gpu.txt`), vLLM >= 0.8.4. Chế độ "thinking" tắt (`enable_thinking=false`).
- Muốn đổi model: sửa `model_id` trong `configs/model.json` (hoặc điền `MODEL_ID` trong notebook). Tổng tham số mọi model dùng lúc inference phải ≤ 15B.

## 8. Điều rút ra từ dữ liệu thật (train 3.000 câu / 30 shop; public test 1.000 câu / 14 shop)

- **Tên thư mục shop khác nhau giữa các tập** (`biz_*` ở train, `dev_*` ở public test) và **rulebook khác nhau** (train: cà phê / nhà thuốc / VLXD; test thêm mẹ & bé, sửa xe, cửa hàng điện thoại) → không hard-code theo shop/ngành, luôn áp rulebook của tập.
- **Rulebook thật ~15–19k ký tự** (không phải 4,5k): `max_rulebook_chars = 20000` cho RTX 5090; trên T4 `datamate/rulebook.py` chỉ giữ các mục ngành khớp README (không chắc thì giữ hết, không đoán).
- **Dữ liệu một shop thường bị tách** (18 sheet theo tháng, `_1`/`_2026`…): `load_business` tự gộp các bảng *cùng cột* thành một bảng, thêm cột `_sheet` (tên sheet gốc – rulebook §8 nói "ngày không ghi năm = năm của sheet") và ghi chú vào README. Bảng mang tên file như trong README (không còn `sheet1`).
- Ô số hay là chữ (`"38,000"`, `"1 bao 25kg"`) → helper `to_num`; khoảng ngày trong profile giờ tính trên **toàn bộ** cột.
- Phân bố đáp án train: number 63 %, percent 19 %, list 6 %, date 5 %, string 3 %, unanswerable 2,3 %, boolean 0,8 %, table 0,6 % → tập trung độ chính xác số/percent; sai một câu −1 nên ngưỡng abstain quan trọng.

## 9. Chạy nhanh nhất và chính xác nhất

| Việc | Tác dụng | Trạng thái |
|---|---|---|
| 1 process / GPU (`predict.sh` tự chia shop thành `--shard i/N` rồi `merge_shards.py`) | ~2× nhanh hơn tensor-parallel với model 8B | có, chưa đo trên GPU thật |
| vLLM + prefix caching (rulebook + README + profile đứng đầu prompt, cùng shop dùng lại) | bỏ ~80 % chi phí nạp prompt | có |
| Bỏ phiếu theo giai đoạn 3→5 lời giải, dừng sớm nếu đồng thuận | tiết kiệm ~40 % thời gian | có |
| Gộp sheet, README/rulebook đầy đủ, `to_num`, profile đúng | ít lỗi code của LLM | có |
| Dựng ngân hàng ví dụ từ **train**: chạy LLM trên train, giữ lời giải cho đáp án khớp đáp án thật (`RUN_MODE = build_exemplars`) | few-shot đúng kiểu câu hỏi của BTC → tăng PA nhiều nhất | có, cần chạy trên GPU |
| Hiệu chỉnh `threshold_answer/unanswerable` bằng `evaluate.calibrate` trên 300–500 câu train giữ lại | tối đa hóa PA (sai −1, bỏ 0) | có, cần chạy trên GPU |
| Đo bằng `train_eval` rồi mới nộp public test (5 lượt/ngày) | tránh đốt lượt | quy trình |

### Chọn 8B hay 14B
- **14B** (mặc định): chính xác hơn, nhưng bf16 ≈ 30 GB nên phải chia 2 GPU (`configs/rtx5090_vllm.json`, `tensor_parallel_size=2`, một tiến trình). Chậm hơn khoảng 2–3 lần so với 8B chạy 2 tiến trình.
- **8B** (`MODEL_ID="Qwen/Qwen3-8B"` + `configs/rtx5090_vllm_8b_2proc.json`): mỗi GPU một tiến trình, nhanh hơn; dùng khi 14B không kịp 6 giờ.
- Kaggle T4 chỉ phù hợp để chạy lát cắt train (`train_eval`) – quá chậm cho 1.000 câu test.
- Quyết định bằng số đo: chạy cùng 300 câu train với cả hai model, so PA và thời gian/câu.

## 10. Tránh overfit và xử lý trường hợp ngoài tập dữ liệu

**Nguyên tắc:** shop test là shop CHƯA THẤY (`dev_*`, có cả ngành mới) → mọi thứ phải dựa vào cấu trúc chung, không dựa vào tên shop/cột/giá trị cụ thể.

| Rủi ro | Cách xử lý (chung, không hard-code theo shop) |
|---|---|
| Đo điểm bị "ảo" vì ví dụ few-shot và câu đánh giá cùng shop | Đánh giá **theo shop**: `HOLDOUT_FOLD="0/5"` giữ ~1/5 số shop; bank `exemplars_safe.jsonl` dựng **không có** các shop đó (`--exclude_fold`); `train_eval` chỉ chạy trên shop giữ lại (`--biz_fold`). Khi nộp bài dùng `EXEMPLAR_SCOPE="all"` |
| Ví dụ few-shot dùng tên cột của shop khác | Prompt ghi rõ "ví dụ từ shop khác – chép cách làm, không chép tên"; chỉ giữ ví dụ mà đáp án khớp đáp án thật |
| Trạng thái đơn viết ~50 kiểu (xong / Đã hoàn thành / completed / hủy đơn / khách hủy / Đổi trả / returned / Đang giao dịch …) | `norm_status()` bằng quy tắc từ khóa (không tra theo shop) + prompt bắt in `crosstab` để kiểm tra; giá trị lạ → `khac` |
| Khách vãng lai viết nhiều kiểu (`Khách lẻ`, `kl`, `lẻ`, rỗng) | Profile liệt kê giá trị hay gặp nhất; prompt yêu cầu tìm hết biến thể |
| Ngày viết lẫn kiểu (`01/01/2025`, `1 thg 1 2025`, `45658`, ISO) | `parse_dates` đọc từng giá trị theo kiểu riêng; đã quét 100 % cột ngày của 44 shop |
| Số lưu dạng chữ (`38,000`, `1.250.000đ`) | `to_num` |
| Dòng "Tổng cộng", "Ghi chú:", tiêu đề lặp | Tự loại + ghi chú vào README gửi cho LLM |
| Quy đổi đơn vị chỉ có trong README (22/44 shop) | Tự parse thành bảng `quy_doi_readme` |
| Dữ liệu chia nhiều sheet/file | Tự gộp, cột `_sheet` |
| Ngành lạ / luật lạ | Luật luôn lấy từ `rulebook.md` của tập; chọn mục ngành theo README nhưng **không đoán** khi README không nêu ngành |
| Câu không trả lời được (kỳ ngoài dữ liệu, shop không ghi, câu chủ quan) | Checklist trong prompt + `data_span()` / `rows_in_period()`; ngưỡng `threshold_unanswerable` hiệu chỉnh trên shop giữ lại |
| LLM sai / lỗi code | Sửa lỗi ≤ 2 lần, bỏ phiếu 3→5; độ đồng thuận thấp → `abstain` (0 điểm thay vì −1) |

**Đã kiểm chứng với đáp án thật (train):** 11 câu viết tay chạy qua đúng pipeline đều khớp (doanh thu, số đơn, % tăng trưởng, top khách, ngày bán đắt nhất, công nợ, bảng top 5 nợ, so sánh dịp Tết). Đây là kiểm tra pipeline/dữ liệu, **không** phải điểm của LLM.
