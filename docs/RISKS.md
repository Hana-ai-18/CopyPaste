# Sổ rủi ro — DataMate pipeline (cập nhật 2026-11-09)

Quy ước trạng thái: **ĐÃ XỬ LÝ** (có code + kiểm thử/đo), **GIẢM NHẸ** (có lưới an toàn nhưng gốc rễ cần GPU/BTC để đóng),
**MỞ** (chưa làm được ở môi trường không GPU — cần người chạy trên máy 5090).

Luật BTC liên quan nhắc lại: chấm PA = (đúng − sai)/N, bỏ trống = 0; ≤ 15B tham số TỔNG mọi model trong image; offline; 2×RTX 5090;
6 giờ; tái lập |ΔPA| ≤ 0,03; luật 6: chỉ tinh chỉnh trên train (KHÔNG dùng câu hỏi/đáp án public/private test để học).

---

## A. Rủi ro "mất trắng" (không có / sai định dạng submission)

| # | Rủi ro | Hậu quả | Xử lý | Trạng thái |
|---|---|---|---|---|
| R1 | Base image `vllm/vllm-openai:v0.6.3` (CUDA 12.1/12.4, torch 2.4) **không có kernel sm_120** cho RTX 5090 → "no kernel image is available" khi nạp model | Không câu nào được LLM trả lời; trước đây = submission rỗng | `docker/Dockerfile`: `ARG BASE_IMAGE` bắt buộc truyền đúng image BTC chỉ định (CUDA ≥ 12.8, torch/vLLM có sm_120); `predict.sh` ghi `logs/env.txt` (torch, arch list, vLLM, tổng tham số). Lưới an toàn: R2 | GIẢM NHẸ — **cần build + chạy thử trên GPU Blackwell** |
| R2 | Model nạp lỗi / OOM / bị kill giữa chừng | Mất toàn bộ hoặc phần lớn đáp án | **Pha tạm**: trước khi nạp model, đường máy tính theo luật (không GPU) trả lời mọi câu nó chắc chắn (spec luật, `tin_cay=cao`, không bị Rule-Delta chặn, không hoà ở biên) → ghi submission ĐỦ mọi id ngay. Nạp model lỗi → giữ file này + `run_summary.json` ghi lỗi. Đã thử: `--llm hf --model /nonexistent` trên 45 câu train → rc=0, submission đủ id, **precision 1,000**, PA 0,756 chỉ với pha tạm | ĐÃ XỬ LÝ |
| R3 | Bị ngắt khi đang ghi → file dở, trùng id, thiếu id | BTC có thể từ chối file | `write_submission()` ghi lại TOÀN BỘ (tmp + fsync + `os.replace` nguyên tử), mỗi lô và mỗi shop của pha tạm; đáp án cuối ghi riêng `answers_final.jsonl` (append + fsync) làm nguồn chạy tiếp. Đã thử chạy tiếp: 20 id, 20 duy nhất, đáp án cũ không đổi | ĐÃ XỬ LÝ |
| R4 | Pha tạm tự nó chạy quá lâu (đọc dữ liệu mọi shop) | Ăn vào 6 giờ | `--provisional_budget` (mặc định 2400 s); dữ liệu shop đọc ở pha tạm được cache → pha chính đọc lại gần như miễn phí | ĐÃ XỬ LÝ |
| R5 | Ngân sách thời gian tính từ lúc vào vòng lặp, quên thời gian nạp model + pha tạm | Vượt 6 giờ | `PROC_T0` lúc import; `TIME_BUDGET=19800` (5,5 h) tính từ đầu tiến trình; hết giờ → câu còn lại = đáp án tạm hoặc abstain | ĐÃ XỬ LÝ |
| R6 | Khoá nội bộ (`_tie`, `_src`…) lọt vào submission | Sai định dạng | `clean_final()` bỏ mọi khoá bắt đầu `_`, chỉ giữ trường hợp lệ theo `type`; áp ở mọi đường ghi. Đã kiểm: 0 dòng có khoá nội bộ | ĐÃ XỬ LÝ |

## B. Rủi ro tái lập (|ΔPA| ≤ 0,03)

| # | Rủi ro | Xử lý | Trạng thái |
|---|---|---|---|
| R7 | "Bỏ bớt bước khi chậm" (shed) quyết định theo **giờ đồng hồ** → máy chấm nhanh/chậm khác ta → khác bước → khác đáp án | `--shed` mặc định **off** (predict.sh `SHED=off`) | ĐÃ XỬ LÝ |
| R8 | Timeout sandbox 15 s: máy chậm hơn → code hết giờ → khác đáp án | `--exec_timeout 40` (nhân theo số bài kiểm) | ĐÃ XỬ LÝ |
| R9 | vLLM không bất biến theo kích thước lô (batch-size variance) dù temperature 0 | TP=1 (không all-reduce giữa 2 GPU), seed cố định, `CUBLAS_WORKSPACE_CONFIG`, `PYTHONHASHSEED=0`. Còn phải **đo flip-rate**: chạy 2 lần cùng image, khác `--batch` (48 vs 16) trên ~500 câu train; nếu PA lệch > 0,01 → bật `VLLM_BATCH_INVARIANT=1` (nếu bản vLLM của image BTC hỗ trợ) hoặc giảm phụ thuộc LLM ở câu sát ngưỡng | MỞ (cần GPU) |
| R10 | Cache shop cũ dùng nhầm cho dữ liệu khác cùng tên thư mục (`biz_101` của train ≠ `biz_101` của private) | Khoá cache gồm chữ ký nội dung: sha1(đường dẫn tương đối:kích thước mọi file) | ĐÃ XỬ LÝ |

## C. Rủi ro đúng/sai đáp án (precision)

| # | Rủi ro | Xử lý | Trạng thái |
|---|---|---|---|
| R11 | Cách viết kỳ lạ không nhận ra: `T5/2026`, `th5/26`, `05/2026`, `5-2026`, `Q2/25`, `quý 2` không năm, `tháng 5` không năm, `tháng 5 năm ngoái`, `từ đầu năm` | `compiler.canon_period_text()` viết lại về `thang M/YYYY` / `quy Q/YYYY` trước khi giải kỳ; năm thiếu theo rulebook (tháng chưa tới → năm trước). Không đụng `T5` trơn (thứ Năm), `ngày 5 tháng 6`, `tháng 2 lần`. Đo bằng `scripts/eval_period_variants.py` (viết lại câu train sang dạng tắt, so đáp án) | ĐÃ XỬ LÝ — đo 150 câu train (5/shop) viết lại sang 7 dạng tắt: code cũ nhận **0/150** ở 6 dạng; code mới cho **đúng y hệt câu gốc** (109 đúng / 3 sai, 3 sai này vốn sai ở câu gốc) ở cả 7 dạng; dạng không ghi năm: 65 đúng / 1 sai / 29 không nhận trên 95 |
| R11b | **Lỗi hồi quy tìm ra khi chạy thử end-to-end**: `quý 2/2026` bị bước "M/YYYY trơn" viết thành `quy thang 2/2026` → trả số liệu THÁNG 2 cho câu hỏi QUÝ (19 câu sai trên 16 shop đầu) | Bước M/YYYY trơn bỏ qua số đứng sau `quy…`/`q`; nhận cả lỗi gõ telex `quys/quyf…`; thêm test `test_quarter_slash_year_not_month` | ĐÃ XỬ LÝ |
| R12 | Lỗi có sẵn: "tháng sau" (tháng kế) bị hiểu thành **tháng 6** sau khi bỏ dấu | Chỉ nhận là tháng 6 khi câu gốc có dấu "tháng sáu" | ĐÃ XỬ LÝ |
| R13 | Đầu tuần cố định Thứ Hai trong compiler dù rulebook có thể đổi | `CMP.WEEK_START = RULE_PARAMS["week_start"]` (tham số đọc từ rulebook) | ĐÃ XỬ LÝ |
| R14 | **Rulebook khác giữa các split** (public: +25 luật mới, −3 luật bỏ, 3 luật mở rộng so với train) → compiler áp công thức train cho luật đã đổi | `rule_delta.py`: so rulebook đang chạy với dấu vân tay rulebook train (`lexicon/rulebook_train_fingerprint.json`, chỉ túi từ, không văn bản). Intent/spec phụ thuộc luật bị **đổi hoặc bỏ** → compiler không được thắng (spec `tin_cay=thap`, sang LLM + đồng thuận). Câu chứa **thuật ngữ mới** → luật mới được ép vào prompt + cặp "luật dễ nhầm" | ĐÃ XỬ LÝ — public: 0 luật đổi, 3 bỏ (giá nhập bình quân, tồn theo đơn vị lớn, giá bán X ngày D). Mô phỏng chạy 3000 câu train với rulebook public: Rule-Delta chặn đúng 75 câu thuộc 2 intent dựa trên luật bị bỏ (`gia_nhap_binh_quan` 61, `ton_don_vi_lon` 14), không chặn intent nào khác (`scripts/eval_compiler.py --rulebook`) |
| R15 | Hoà ở biên top-k (BTC loại câu hoà khỏi bộ đề, nên nếu ta thấy hoà thì gần như chắc ta tính sai) | "Tie oracle": compiler đánh dấu `_tie` khi giá trị thứ k = thứ k+1 → **bỏ trống** (không cho LLM ghi đè) | ĐÃ XỬ LÝ — trên train: **0** câu bị đánh dấu hoà (khớp việc BTC loại câu hoà), nên không mất câu đúng nào |
| R16 | Tên trả lời lệch chính tả so với dữ liệu (thiếu dấu, thừa khoảng trắng) | `snap_final()`: chuẩn hoá theo đúng hàm chuẩn hoá của BTC (NFC, thường, ký tự không chữ-số → khoảng trắng) rồi khớp về giá trị có thật trong cột; không khớp → đặc trưng `snap_miss` cho mô hình bỏ trống | ĐÃ XỬ LÝ |
| R17 | Hàm chấm nội bộ khác hàm chấm BTC → đo sai | `_snorm` = đúng mô tả chuẩn hoá BTC | ĐÃ XỬ LÝ |
| R18 | pandas 3: cột chữ có dtype `string` chứ không phải `object` → các nhánh `is_object_dtype` bỏ sót | Dùng `is_object_dtype or is_string_dtype` (clean_table, _vocab) | ĐÃ XỬ LÝ — image BTC có thể dùng pandas 2: cả hai đều chạy |
| R19 | Ngưỡng bỏ trống fit trên train bị lạc quan vì chia fold theo shop mà shop cùng "họ dữ liệu" (generator template) rò rỉ | `fit_abstain.py --group family` (mặc định): fold theo họ dữ liệu (`data_family`: tên file bỏ số) | ĐÃ XỬ LÝ (code) — **cần fit lại trên log train chạy có GPU** |
| R20 | Dữ liệu shop bẩn (tồn kho không khớp nhập − bán, tiền đơn ≠ tổng dòng) → công thức đúng vẫn ra số "sai" so với generator | `shop_quality()` đo 2 chỉ số, đưa vào log + đặc trưng bỏ trống `shop_ton_lech` | GIẢM NHẸ |

## D. Tuân thủ luật

| # | Rủi ro | Xử lý | Trạng thái |
|---|---|---|---|
| R21 | Vượt 15B khi đóng gói thêm bge-m3 / PhoGPT cạnh LLM chính | `scripts/count_params.py` đếm safetensors mọi model trong `/models`, exit 1 nếu > 15B; chạy ngay trong `docker build` | ĐÃ XỬ LÝ |
| R22 | Vô tình học từ public/private test (luật 6) | Không code nào đọc `answers` ngoài split train; `spec_bank`, fingerprint, abstain model chỉ dựng từ train; Rule-Delta chỉ **so sánh** rulebook (đầu vào hợp lệ lúc suy luận), không học tham số từ test | ĐÃ XỬ LÝ |
| R23 | Đẩy dữ liệu BTC lên git | `lexicon/spec_bank.jsonl` chứa **câu hỏi train** → vẫn bị `.gitignore` (bản trước có ý định mở, đã thu hồi). Docker lấy từ thư mục làm việc qua `.dockerignore` (cho phép riêng file này); build **thất bại** nếu thiếu file | ĐÃ XỬ LÝ |
| R24 | `COPY . /app` chép cả `models/` lần hai, cache, outputs, có thể cả dữ liệu thi vào image | Chỉ `COPY *.py lexicon scripts`; thêm `.dockerignore` loại data/outputs/cache/zip/answers | ĐÃ XỬ LÝ |
| R25 | Nâng cấp `transformers` trong image vLLM làm hỏng vLLM dựng sẵn | Dockerfile không nâng cấp transformers/torch; chỉ cài gói thuần Python | ĐÃ XỬ LÝ |
| R26 | ViLexNorm (CC BY-NC-SA 4.0): dùng làm tài nguyên công khai được luật 6 cho phép, nhưng bảng ánh xạ của nó **xung đột miền** (kg→không, bn→bạn, k→không) | Chỉ dùng để ĐO bộ sửa chữ (`scripts/eval_vilexnorm.py`), **không** gộp vào normalizer | ĐÃ XỬ LÝ |

## F. Chạy trên Kaggle (phát triển + nộp public)

| # | Rủi ro | Xử lý | Trạng thái |
|---|---|---|---|
| R27 | Cache pickle shop dựng bằng pandas 3 nạp ở pandas 2 (hoặc file pickle ghi dở) → lỗi / sai ngầm | Khoá cache thêm phiên bản pandas; nạp lỗi → đọc lại dữ liệu gốc; ghi pickle nguyên tử | ĐÃ XỬ LÝ |
| R28 | `pip install vllm` bản mới nhất (0.33, torch 2.14) có thể không chạy trên T4 / driver Kaggle; vLLM ≥ 0.11 bỏ engine V0 (đường chạy quen thuộc của T4) | Notebook ghim `vllm==0.10.1.1` (+ transformers cùng thời), `VLLM_USE_V1=0`; nạp THẬT model + sinh thử trước khi chạy; lỗi thì thử `0.8.5.post1`, rồi `hf`, rồi không LLM | ĐÃ XỬ LÝ (chưa chạy được trên T4 thật ở môi trường này) |
| R29 | Không LLM nào chạy được → dummy LLM sinh đáp án vô nghĩa | `--llm none`: chỉ nộp đáp án chắc chắn của đường máy tính (precision 1,0 trên mẫu train), còn lại abstain | ĐÃ XỬ LÝ |
| R30 | Hết 12 giờ phiên Kaggle khi chưa xong public | Ngân sách public = thời gian còn lại − 15 phút; submission luôn đủ câu; chạy lại ô bước 8 sẽ tiếp tục | ĐÃ XỬ LÝ |
| R32 | Đổi sang **Qwen3.5-9B** (Apache-2.0, phát hành 24/02/2026, ~9,7B tham số kể cả bộ mã hoá ảnh → vẫn < 15B): (a) mặc định "nghĩ" trước khi trả lời → hết max_tokens trước khi viết code; (b) cần vLLM ≥ 0.17 (chỉ engine V1) — chưa ai trong nhóm chạy trên T4; (c) trọng số BF16, T4 phải chạy fp16 (~19 GB → bắt buộc TP=2); (d) là model ảnh+chữ; (e) kiến trúc lai Gated DeltaNet: prefix caching có thể không hỗ trợ → chậm hơn | (a) `enable_thinking=False` trong chat template + bỏ `<think>…</think>` ở đầu ra; (b) notebook ghim vLLM 0.19.1 / 0.17.1 (CUDA 12.8), không dùng ≥ 0.20 (CUDA 13); (c)(d) `limit_mm_per_prompt` ảnh = 0, TP=2; (e) tự thử bớt prefix caching nếu bản vLLM từ chối. Nạp thử bằng CHÍNH lớp LLM của pipeline và đòi trả lời đúng `17+25=42` (bắt fp16 ra rác / vẫn đang nghĩ). Không chạy được → **tự lùi về Qwen2.5-Coder-7B-AWQ**. Chất lượng 9B so với Coder-7B trên bài này: **chưa đo**. Chạy thật trên Kaggle T4 (10/11): vLLM 0.19.1 TP=2 lỗi 'Engine core initialization failed' (lỗi gốc nằm trong tiến trình con, bị cắt khỏi log) → đã sửa: lỗi khởi tạo engine V1 (RuntimeError) cũng được thử cấu hình nhẹ hơn (tắt prefix caching, rồi enforce_eager); notebook lưu log đầy đủ `probe_logs/` và in dòng lỗi gốc **Lỗi gốc đã tìm ra** (log Kaggle: dòng 'non-default args' in 2 lần): script nạp thử thiếu `if __name__ == "__main__"` → vLLM mở tiến trình con bằng spawn, tiến trình con chạy lại script và tạo LLM lồng nhau. Đã thêm khối bảo vệ + test; tái hiện được lỗi bằng vLLM giả và xác nhận bản sửa qua. Pipeline chính vốn đã có khối này | GIẢM NHẸ — chờ chạy lại trên T4 |
| R31 | Kết quả khác giữa Python 3.11/pandas 2.2 (Kaggle) và Python 3.13/pandas 3 (máy dev) | Đo: đường máy tính trên 3000 câu train cho **0 khác biệt** đáp án giữa hai môi trường | ĐÃ KIỂM |

## G. Lần chạy Kaggle 10/11 (Qwen3.5 lỗi → hf → 621 câu bỏ trống)

| # | Rủi ro / lỗi thật | Xử lý | Trạng thái |
|---|---|---|---|
| R33 | T4 (sm75): vLLM 0.19.1 chọn backend **FlashInfer**, phải biên dịch kernel prefill head_dim 256 lúc chạy → `ninja`/`ld` lỗi | GPU < sm80: ép `attention_config={"backend": "TRITON_ATTN"}` (dự phòng FLEX_ATTENTION), tắt bộ lấy mẫu FlashInfer | GIẢM NHẸ — chờ chạy lại T4 |
| R34 | Cài vLLM bản khác sau bản đầu → `flashinfer-cubin 0.6.6 ≠ flashinfer 0.6.4` | Gỡ sạch `vllm flashinfer-*` trước khi cài bản khác; không cài lại nếu đúng bản | ĐÃ XỬ LÝ |
| R35 | Kaggle đã lên **Python 3.13** → vLLM < 0.10.2 không cài được → model dự phòng Qwen2.5-AWQ không có đường vLLM → rơi xuống hf | Mỗi bản vLLM thử cả `MODEL` lẫn `FALLBACK_MODEL` (Qwen2.5-AWQ chạy trên vLLM 0.19/0.17); bỏ bản cũ khi Python ≥ 3.13 | ĐÃ XỬ LÝ |
| R36 | Backend hf: attention 'eager' + lô 4 prompt ~10k token → OOM 11–35 GiB → **cả lô 'fatal'** → mọi câu chỉ còn đáp án máy tính | `attn_implementation="sdpa"`; LLM sinh lỗi cả lô → sinh lẻ; prompt lỗi → rỗng (không làm hỏng câu khác); hf chạy 1 đường | ĐÃ XỬ LÝ |
| R37 | Đáp án máy tính "chắc chắn" sai trên dữ liệu chưa thấy (public: 46 sai / 379 trả lời): bộ lọc bị bỏ sót (món viết khác tên shop, mã NV, nhóm hàng), 'hôm nào/bữa nào' bị hiểu là tổng, 'top năm' = 1, 'ngày mô quán' bị coi là hỏi địa chỉ, nợ quá N ngày bị trả bảng, ngành mới (điện thoại, sửa xe, mẹ và bé) | Sửa nhận dạng theo LOẠI câu; chặn chữ lạ (từ vựng train + tên thực thể shop); ngành không có ở train → hạ tin cậy; 'không thấy dữ liệu' → hạ tin cậy. Mọi thay đổi đo lại trên TRAIN (không giảm câu đúng) | ĐÃ XỬ LÝ — số đo trong báo cáo |
| R39 | Bỏ trống quá nhiều: luật mặc định chỉ trả lời khi 2 đường cùng đáp án; câu 2 đường lệch / chỉ 1 đường / máy tính tin cậy thấp đều bỏ trống dù đáp án có thể đúng > 50% | **Cứu theo hiệu chỉnh**: câu bỏ trống luôn ghi ứng viên tốt nhất + loại; `calibrate_decisions.py` đo trên 6 shop TRAIN tỉ lệ đúng từng loại (và loại\|intent); loại nào cận dưới Wilson > 0,5 (đủ ≥ 12 mẫu) thì ở public trả lời bằng ứng viên. Notebook bật mặc định (`RUN_TRAIN_EVAL = True`) | CHỜ chạy có LLM để biết cứu được bao nhiêu |
| R40 | Hiệu chỉnh trên train lạc quan: ví dụ few-shot lấy từ spec_bank có thể là câu của CHÍNH shop đang chạy (thậm chí cùng câu) | Truy hồi ví dụ bỏ mọi ví dụ cùng shop (`exclude_biz`) | ĐÃ XỬ LÝ |
| R38 | Luật 6: sửa code sau khi so với đáp án public | Chỉ sửa LỖI TỔNG QUÁT (theo dạng câu), không thêm luật cho câu/shop cụ thể, không chỉnh ngưỡng theo public; từ vựng chỉ dựng từ train; kiểm lại trên train | LƯU Ý cho nhóm |

## Số đo tổng (đường máy tính, không LLM, 3000 câu train, pandas 3.0.5)

| | trả lời | đúng | sai | precision | PA |
|---|---|---|---|---|---|
| Code nhận vào | 1807 | 1798 | 9 | 0,9950 | 0,5963 |
| Code sau sửa | **1923** | **1914** | **9** | 0,9953 | **0,6350** |
| — chỉ intent tin cậy cao (cũ → mới) | 1763 → 1879 | 1762 → 1878 | 1 → 1 | 0,9995 | 0,5870 → 0,6257 |

+116 câu đúng, 0 câu sai thêm: đến từ quý (`quý 2/2026`, `quý 3 năm 2025`), `từ đầu năm`, cách viết tháng tắt.

## E. Việc còn MỞ (cần máy GPU, theo thứ tự ưu tiên)

1. **R1**: build với base image BTC, chạy `selftest` + 50 câu train trên GPU Blackwell thật; xem `logs/env.txt` có `sm_120`.
2. **R9**: đo flip-rate 2 lần chạy × 2 kích thước lô.
3. Chạy toàn bộ train có LLM → `scripts/fit_abstain.py --group family` → `lexicon/abstain_model.json`; `scripts/calibrate_decisions.py` → `lexicon/decision_calibration.json` (hai file này hiện CHƯA có, predict.sh tự bỏ qua nếu thiếu).
4. So model chính: Qwen2.5-Coder-7B vs model ≤ 14B mới hơn trên cùng bộ train (PA, thời gian, VRAM).
5. Cân nhắc DP=2 (2 tiến trình vLLM, mỗi GPU một nửa số shop) thay vì TP — nhanh gấp ~2 và không có all-reduce; cần tách theo shop để vẫn tái lập.
