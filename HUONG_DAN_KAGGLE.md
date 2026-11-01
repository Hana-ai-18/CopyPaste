# Chạy DataMate trên Kaggle (GPU) – hướng dẫn đầy đủ

## 0. Kaggle làm được gì, không làm được gì
| Việc | Kaggle? |
|---|---|
| Thử pipeline, đo điểm trên train, dựng ví dụ few-shot, tạo `submission.jsonl` để nộp **public test** | **Được** (GPU T4, phiên tối đa 12 giờ; hạn mức GPU/tuần thường khoảng 30 giờ – xem trong tài khoản của bạn) |
| Build Docker để nộp bài cuối (hạn 16/11) | **Không** (Kaggle không có Docker). Cần một máy có Docker + GPU (thuê theo giờ), xem mục 7 |
| Chạy private test | Do BTC chạy Docker của bạn trên 2×RTX 5090 |

Chọn GPU **T4 ×2** (không dùng P100: bitsandbytes 4-bit không ổn định). Notebook tự chạy **1 tiến trình / GPU** (mỗi T4 một nửa số shop) nên nhanh gấp đôi.

## 1. Chuẩn bị (một lần)
1. **GitHub:** tạo repo **private**, giải nén `zaic2026-datamate.zip`, `git init && git add -A && git commit && git push`. (`.gitignore` đã loại dữ liệu, model, ví dụ few-shot.)
2. **Dữ liệu:** trên Kaggle → *Datasets → New dataset* → tải `DATAMATE.zip` (hoặc `train.zip` + `public_test.zip`), đặt **Private** (thể lệ cấm phát tán dữ liệu). Notebook tự giải nén zip lồng nhau.
3. **Secrets** (Notebook → *Add-ons → Secrets*): `GITHUB_TOKEN` (token đọc repo private). `HF_TOKEN` chỉ cần nếu tải model bị giới hạn.
4. **Cài đặt notebook:** Upload `datamate_kaggle.ipynb` → *Settings*: **Accelerator = GPU T4 ×2**, **Internet = On**, *Persistence = Files only*; *Add data* → chọn dataset ở bước 2.

## 2. Ô cấu hình (ô code đầu tiên)
```
GITHUB_REPO   = "https://github.com/<bạn>/<repo>.git"
MODEL_ID      = "Qwen/Qwen3-8B"          # nhanh.   Tốt hơn: "" (= Qwen3-14B theo configs/model.json)
CONFIG        = "configs/t4_kaggle.json" # với 14B: "configs/t4_kaggle_14b.json"
HOLDOUT_FOLD  = "0/5"   EXEMPLAR_N = 400   EVAL_N = 150
NUM_PROCS     = 0       # 0 = tự đếm GPU (T4×2 -> 2 tiến trình)
RESUME        = True
```

## 3. Thứ tự chạy (mỗi lần đổi `RUN_MODE`, chạy lại từ đầu notebook)
| # | `RUN_MODE` | Mục đích | Ghi chú |
|---|---|---|---|
| 1 | `smoke` | kiểm tra môi trường, không cần GPU | vài phút, phải in `SMOKE TEST OK` |
| 2 | `train_eval` với `EVAL_N=20` | lần đầu chạy LLM thật: xem model nạp được không, thời gian/câu, lỗi code | chưa có bank → chạy zero-shot |
| 3 | `build_exemplars` (`EXEMPLAR_SCOPE="safe"`) | LLM giải ~400 câu train (ngoài các shop giữ lại), giữ lời giải **đúng** làm ví dụ | tạo `exemplars/exemplars_safe.jsonl` |
| 4 | `train_eval` với `EVAL_N=150` | **điểm trung thực** trên shop chưa thấy + gợi ý ngưỡng `threshold_answer/unanswerable` | điểm này gần với private nhất |
| 5 | `build_exemplars` với `EXEMPLAR_SCOPE="all"` | bank cuối từ mọi shop train → `exemplars/exemplars.jsonl` | |
| 6 | `public_test` | tạo `/kaggle/working/submission.jsonl` | tải về, nộp ở mục Submission (5 lượt/ngày) |

Chưa đo tốc độ trên GPU thật. Ước lượng thô: 8B 4-bit ≈ 20–40 giây/câu/GPU → 1.000 câu / 2 GPU ≈ 3–6 giờ; 14B chậm hơn ~1,5–2 lần (có thể vượt 12 giờ → dùng `RESUME` hoặc bỏ bớt `stages`). **Sau bước 2 hãy tính lại** từ thời gian thực in ra.

## 4. Giữ kết quả giữa các phiên
- `/kaggle/working` chỉ còn khi bạn **Save Version** (Save & Run All để chạy nền tới 12 giờ mà không cần mở trình duyệt). Tải `exemplars_safe.jsonl`/`exemplars.jsonl` từ *Output*, rồi tải lên một **dataset private** (ví dụ `datamate-exemplars`); lần sau notebook tự nhập chúng.
- Phiên bị ngắt giữa chừng: giữ `RESUME = True` và chạy lại – các câu đã trả lời được giữ nguyên.

## 5. Lỗi thường gặp
| Hiện tượng | Cách xử lý |
|---|---|
| `CUDA out of memory` | `--set hf.batch_size=1`, giảm `hf.max_input_tokens` (12000 → 9000) trong file config |
| Tràn RAM khi nạp model (2 tiến trình) | tăng `STAGGER_S` (90 → 180) hoặc đặt `NUM_PROCS = 1` |
| Tải model chậm / lỗi mạng | kiểm tra *Internet = On*; chạy lại (script bỏ qua file đã tải) |
| `STOP: the model has ... B parameters` | model vượt 15B – đổi `MODEL_ID` |
| Kết quả khác nhau giữa 2 lần chạy | bình thường ở mức nhỏ; thể lệ cho phép lệch PA ≤ 0,03 – so hai lần chạy public test |

## 6. Kiểm tra trước khi nộp public test
`scripts/validate_submission.py` chạy tự động trong notebook (đủ id, đúng type, đúng định dạng). Không nộp nếu nó báo lỗi.

## 7. Docker cho bài nộp cuối (hạn 16/11 23:59)
Cần máy Linux có Docker + NVIDIA Container Toolkit + 2 GPU (thuê theo giờ). Các lệnh nằm trong README mục 4. Điểm cần nhớ: model 14B ở bf16 phải chia 2 GPU (`configs/rtx5090_vllm.json`); chạy thử `docker run --network none` trên một lát public test và so điểm với Kaggle trước khi nộp. `BASE_IMAGE` lấy theo hướng dẫn Docker của BTC.

---

## 8. Chạy 14B: tránh tràn RAM/VRAM và chạy tiếp (resume)

### Bộ nhớ (ước tính, chưa đo trên T4 thật)
| Thành phần | 14B 4-bit trên 1 T4 (15 GB VRAM) |
|---|---|
| Trọng số (nf4 + embedding/lm_head fp16) | ≈ 10 GB |
| KV cache 1 câu ≈ 12k token | ≈ 2 GB |
| Activation + CUDA context | ≈ 1–1,5 GB |
| **Tổng** | ≈ 13–13,5 GB → chỉ vừa với **batch 1** (`configs/t4_kaggle_14b.json`) |

Đã làm sẵn trong code:
- Mỗi tiến trình ghim **toàn bộ model vào 1 GPU** (`device_map={"":0}`) → không bao giờ âm thầm đẩy lớp sang CPU (chậm + tràn RAM); không đủ chỗ thì báo OOM rõ ràng.
- Nạp model **từng shard** (`low_cpu_mem_usage`) → RAM đỉnh ≈ 1 shard (~4 GB), không phải 30 GB. Hai tiến trình nạp cách nhau `STAGGER_S` giây.
- Gặp **CUDA OOM** giữa chừng: giải phóng cache, thử lại từng prompt với ngữ cảnh ngắn hơn 25 % thay vì bỏ câu.
- Nếu vẫn OOM: giảm `hf.max_input_tokens` (12000 → 9000) và `max_rulebook_chars` trong `configs/t4_kaggle_14b.json`; hoặc dùng 8B.

### Phương án chạy tốt nhất
1. `MODEL_ID=""`, `CONFIG="configs/t4_kaggle_14b.json"`, `NUM_PROCS=0` (T4 ×2 → 2 tiến trình, mỗi tiến trình 1 GPU, mỗi GPU giữ 1 bản 14B).
2. Chạy bằng **Save & Run All** (chạy nền, không cần mở trình duyệt).
3. Với `SESSION_HOURS = 11`, agent tự tính ngân sách thời gian còn lại và **dừng đúng hạn** (chừa 30 phút) → gộp và lưu kết quả rồi kết thúc êm, không bị Kaggle ngắt giữa chừng.
4. Câu nào chưa kịp trả lời được ghi `abstain` tạm trong file nộp và đánh dấu vào `logs/unfinished.json`.

### Chạy tiếp ở phiên sau
1. Sau khi phiên 1 xong: **Save Version**. 
2. Phiên 2: mở notebook, *Add Data → Notebook Output* → chọn chính notebook ở phiên 1. Notebook tự chép `public_test_out/` (hoặc `train_eval/`, `exemplar_run/`) về `/kaggle/working`.
3. Giữ `RESUME = True`, chạy lại: các câu đã trả lời được giữ, các câu trong `unfinished.json` (hết giờ / OOM) được làm tiếp. Xong thì tải `submission.jsonl`.
4. Mất kết nối giữa chừng (trình duyệt đóng): với Save & Run All phiên vẫn chạy; nếu phiên bị ngắt cứng, kết quả đã ghi tới từng câu (file được flush sau mỗi câu) – nhưng chỉ lấy được nếu Kaggle lưu output của phiên bị ngắt; vì vậy **đừng đặt `SESSION_HOURS` sát 12**.

### Mẹo để vừa nhanh vừa tốt
- Làm bước `train_eval` nhỏ (20 câu) với 14B trước để lấy **giây/câu thật**, rồi tính: `số_câu × giây/câu ÷ số_GPU`. Nếu > 10 giờ: giảm `agent.stages` từ `[3,5]` sang `[3]` (giảm ~40 % thời gian), hoặc dùng 8B cho phần lớn và 14B chỉ cho lần chạy cuối.
- Chạy `public_test` theo lát (`--offset/--limit`) nếu muốn chia nhiều phiên rõ ràng.
- Không đổi `seed`, config, model giữa hai phiên của cùng một lần nộp (đảm bảo tái lập).

---

## 9. Bước PREFLIGHT – đo thật thay vì đoán (làm trước khi chạy dài)

`RUN_MODE = "preflight"` chạy `PREFLIGHT_N` (6) câu train thật trên GPU 0, rồi in và lưu `auto_tune.json`:
- thời gian nạp model, giây/câu, **VRAM đỉnh**, **RAM đỉnh**;
- **dự báo số giờ** cho cả lần chạy (số câu của public test ÷ số GPU);
- đề xuất tự động: nếu VRAM sát trần (>93 %) hoặc có OOM → giảm `max_input_tokens` còn 9000 và `batch_size=1`; nếu dự báo vượt ngân sách giờ → `stages=[3]` (hoặc `[1]`); nếu vẫn không kịp → khuyên dùng 8B / chia nhiều phiên.

Với `AUTO_TUNE = True`, các lần chạy sau (`train_eval`, `build_exemplars`, `public_test`) **tự áp dụng** các đề xuất đó.

### Bảng rủi ro và cách chặn
| Rủi ro | Cách chặn (đã làm trong code) | Còn lại phải làm tay |
|---|---|---|
| Tràn VRAM với 14B | preflight đo + tự hạ context; OOM giữa chừng → thử lại từng prompt ngắn hơn 25 % | nếu preflight báo "Even one sample…/OOM mãi" → đổi sang 8B |
| Tràn RAM khi nạp | nạp từng shard, ghim 1 GPU/tiến trình, giãn cách nạp | chỉ khi vẫn tràn: `NUM_PROCS=1` |
| Không kịp giờ | dự báo từ số đo thật; tự giảm số lượt bỏ phiếu; trong lúc chạy tự chuyển "fast mode" khi thiếu giờ; dừng đúng hạn | — |
| Phiên Kaggle bị ngắt | dừng êm trước giới hạn; mỗi câu ghi ngay; `unfinished.json`; nhập lại output phiên trước | đặt `SESSION_HOURS` ≤ 11 |
| Model > 15B | `model_check` đếm tham số thật khi tải và khi chạy, dừng nếu vượt | tự xem model card: giấy phép + ngày phát hành < 01/10/2026 (30 giây) |
| Điểm LLM chưa biết | `train_eval` trên shop giữ lại + hiệu chỉnh ngưỡng bỏ trống | chạy bước này và đọc điểm |
| Đóng gói Docker | `scripts/docker_selftest.sh` (build + chạy `--network none` + validate, không cần GPU/model) | chạy ở máy có Docker; rồi chạy thật 1 lát public test trên máy GPU |

Mình đã chạy `predict.sh` đúng như container sẽ chạy (môi trường offline, mock LLM) ở cả chế độ 1 và 2 tiến trình: kết quả giống hệt nhau và qua `validate_submission`. Mình **không** build được Docker thật (môi trường của mình không có Docker daemon).
